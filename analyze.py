#!/usr/bin/env python3
"""Reproduce the delivery-QC analysis from the supplied CSV."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


REJECTION_HOURS = 75
EARLY_CHECK_HOURS = 20
SPLIT_DATE = "2026-09-06"
LOCAL_TIMEZONE = "Asia/Shanghai"


def metrics(frame: pd.DataFrame, flagged: pd.Series) -> dict[str, int | float | None]:
    failed = frame["status_ok"].eq(0)
    flag_count = int(flagged.sum())
    true_positive = int((flagged & failed).sum())
    false_positive = flag_count - true_positive
    false_negative = int((~flagged & failed).sum())
    true_negative = int((~flagged & ~failed).sum())
    failure_count = int(failed.sum())
    return {
        "labeled_batches": int(len(frame)),
        "failures": failure_count,
        "baseline_failure_rate": round(failure_count / len(frame), 4) if len(frame) else None,
        "flagged_batches": flag_count,
        "flagged_failures": true_positive,
        "flagged_passes": false_positive,
        "unflagged_failures": false_negative,
        "unflagged_passes": true_negative,
        "precision": round(true_positive / flag_count, 4) if flag_count else None,
        "failure_recall": round(true_positive / failure_count, 4) if failure_count else None,
        "theoretical_net_hours": true_positive * REJECTION_HOURS - flag_count * EARLY_CHECK_HOURS,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("doc/delivery_batches.csv"))
    parser.add_argument("--output", type=Path, default=Path("analysis_results.json"))
    args = parser.parse_args()

    raw = pd.read_csv(args.input)
    required = {
        "batch_id",
        "packaged_at",
        "calib_synced_at",
        "fleet_base",
        "sensor_modality",
        "valid_frame_ratio",
        "package_size_mb",
        "status_ok",
    }
    missing_columns = sorted(required - set(raw.columns))
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    exact_duplicate_rows = int(raw.duplicated().sum())
    data = raw.drop_duplicates().copy()
    data["packaged_utc"] = pd.to_datetime(data["packaged_at"], utc=True)
    data["calib_utc"] = pd.to_datetime(data["calib_synced_at"], utc=True)
    data["calib_age_minutes"] = (
        data["packaged_utc"] - data["calib_utc"]
    ).dt.total_seconds() / 60
    data["packaged_local"] = data["packaged_utc"].dt.tz_convert(LOCAL_TIMEZONE)
    data["local_day"] = data["packaged_local"].dt.strftime("%Y-%m-%d")

    if data["calib_age_minutes"].lt(0).any():
        raise ValueError("Found calibration timestamps later than packaging timestamps")
    observed_statuses = set(data["status_ok"].dropna().unique())
    if not observed_statuses.issubset({0, 1}):
        raise ValueError(f"Unexpected status_ok values: {sorted(observed_statuses)}")
    if data["batch_id"].duplicated().any():
        raise ValueError("batch_id is still duplicated after exact-row deduplication")

    labeled = data[data["status_ok"].notna()].copy()
    train = labeled[labeled["local_day"] < SPLIT_DATE].copy()
    holdout = labeled[labeled["local_day"] >= SPLIT_DATE].copy()
    if train.empty or holdout.empty:
        raise ValueError("The chronological training or holdout period is empty")

    # Choose the threshold using only the earlier, labeled batches. Each observed
    # training age is a candidate. Ties prefer the higher cutoff (fewer alerts).
    candidates: list[tuple[int, float]] = []
    train_failed = train["status_ok"].eq(0)
    for threshold in sorted(train["calib_age_minutes"].unique()):
        flagged = train["calib_age_minutes"].ge(threshold)
        net_hours = (
            int((flagged & train_failed).sum()) * REJECTION_HOURS
            - int(flagged.sum()) * EARLY_CHECK_HOURS
        )
        candidates.append((net_hours, float(threshold)))
    best_net, cutoff_minutes = max(candidates, key=lambda item: (item[0], item[1]))

    age_rule = lambda frame: frame["calib_age_minutes"].ge(cutoff_minutes)
    frame_rule = lambda frame: frame["valid_frame_ratio"].le(2)
    fresh = labeled["calib_age_minutes"].lt(cutoff_minutes)
    stale = ~fresh

    age_minutes = data["calib_age_minutes"]
    result = {
        "source": str(args.input),
        "time_zone_for_split": LOCAL_TIMEZONE,
        "data_quality": {
            "raw_rows": int(len(raw)),
            "exact_duplicate_rows_removed": exact_duplicate_rows,
            "unique_rows": int(len(data)),
            "unique_batch_ids": int(data["batch_id"].nunique()),
            "pending_rows_excluded_from_failure_rates": int(data["status_ok"].isna().sum()),
            "labeled_rows": int(len(labeled)),
            "failures": int(labeled["status_ok"].eq(0).sum()),
            "passes": int(labeled["status_ok"].eq(1).sum()),
            "negative_calibration_ages": int(age_minutes.lt(0).sum()),
            "packaged_at_local_range": [
                data["packaged_local"].min().isoformat(),
                data["packaged_local"].max().isoformat(),
            ],
        },
        "cost_assumptions": {
            "hours_per_final_rejection": REJECTION_HOURS,
            "hours_per_early_check": EARLY_CHECK_HOURS,
            "break_even_flagged_failure_rate": round(EARLY_CHECK_HOURS / REJECTION_HOURS, 4),
            "net_hours_assumes_every_flagged_failure_is_prevented": True,
        },
        "split": {
            "training_local_dates": f"through {SPLIT_DATE} exclusive",
            "holdout_local_dates": f"from {SPLIT_DATE} inclusive",
            "training_batches": int(len(train)),
            "holdout_batches": int(len(holdout)),
        },
        "selected_rule": {
            "expression": f"calibration age >= {cutoff_minutes:g} minutes",
            "threshold_selected_on_training_net_hours": int(best_net),
            "training": metrics(train, age_rule(train)),
            "chronological_holdout": metrics(holdout, age_rule(holdout)),
            "all_labeled_historical_data": metrics(labeled, age_rule(labeled)),
        },
        "risk_by_calibration_age": {
            "below_threshold": {
                "batches": int(fresh.sum()),
                "failures": int((fresh & labeled["status_ok"].eq(0)).sum()),
                "failure_rate": round(
                    float(labeled.loc[fresh, "status_ok"].eq(0).mean()), 4
                ),
            },
            "at_or_above_threshold": {
                "batches": int(stale.sum()),
                "failures": int((stale & labeled["status_ok"].eq(0)).sum()),
                "failure_rate": round(
                    float(labeled.loc[stale, "status_ok"].eq(0).mean()), 4
                ),
            },
        },
        "secondary_signal_valid_frame_ratio_le_2": {
            "training": metrics(train, frame_rule(train)),
            "chronological_holdout": metrics(holdout, frame_rule(holdout)),
            "all_labeled_historical_data": metrics(labeled, frame_rule(labeled)),
        },
        "outcome_rates_by_fleet": {
            str(group): {
                "labeled_batches": int(len(part)),
                "failures": int(part["status_ok"].eq(0).sum()),
                "failure_rate": round(float(part["status_ok"].eq(0).mean()), 4),
            }
            for group, part in labeled.groupby("fleet_base", sort=True)
        },
        "outcome_rates_by_modality": {
            str(group): {
                "labeled_batches": int(len(part)),
                "failures": int(part["status_ok"].eq(0).sum()),
                "failure_rate": round(float(part["status_ok"].eq(0).mean()), 4),
            }
            for group, part in labeled.groupby("sensor_modality", sort=True)
        },
        "mean_package_size_mb": {
            "failed": round(float(labeled.loc[labeled["status_ok"].eq(0), "package_size_mb"].mean()), 2),
            "passed": round(float(labeled.loc[labeled["status_ok"].eq(1), "package_size_mb"].mean()), 2),
        },
    }

    rendered = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
