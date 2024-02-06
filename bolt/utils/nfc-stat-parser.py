#!/usr/bin/env python3
import argparse
import csv
import re
import sys
import os
from statistics import geometric_mean

TIMING_LOG_RE = re.compile(r"(.*)/(.*).tmp(.*)")


def main():
    parser = argparse.ArgumentParser(
        description="BOLT NFC stat parser",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "input", nargs="+", help="timing.log files produced by llvm-bolt-wrapper"
    )
    parser.add_argument(
        "--check_longer_than",
        default=2,
        type=float,
        help="Only warn on tests longer than X seconds for at least one side",
    )
    parser.add_argument(
        "--threshold_single",
        default=10,
        type=float,
        help="Threshold for a single test result swing, abs percent",
    ),
    parser.add_argument(
        "--threshold_agg",
        default=5,
        type=float,
        help="Threshold for geomean test results swing, abs percent",
    ),
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    def fmt_delta(value, exc_threshold, above_bound=True):
        formatted_value = format(value, "+.2%")
        if not above_bound:
            formatted_value += "?"
        elif exc_threshold and sys.stdout.isatty():  # terminal supports colors
            return f"\033[1m{formatted_value}\033[0m"
        return formatted_value

    # Ratios for geomean computation
    time_ratios = []
    mem_ratios = []
    # Whether any test exceeds the single test threshold (mem or time)
    threshold_single = False
    # Whether geomean exceeds aggregate test threshold (mem or time)
    threshold_agg = False

    if args.verbose:
        print(f"# Individual test threshold: +-{args.threshold_single}%")
        print(f"# Aggregate (geomean) test threshold: +-{args.threshold_agg}%")
        print(
            f"# Checking time swings for tests with runtime >"
            f"{args.check_longer_than}s - otherwise marked as ?"
        )
        print("Test/binary BOLT_wall_time BOLT_max_rss")

    for input_file in args.input:
        input_dir = os.path.dirname(input_file)
        with open(input_file) as timing_file:
            timing_reader = csv.reader(timing_file, delimiter=";")
            for row in timing_reader:
                test_name = row[0]
                m = TIMING_LOG_RE.match(row[0])
                if m:
                    test_name = f"{input_dir}/{m.groups()[1]}/{m.groups()[2]}"
                else:
                    # Prepend input dir to unparsed test name
                    test_name = input_dir + "#" + test_name
                time_a, time_b = float(row[1]), float(row[3])
                mem_a, mem_b = int(row[2]), int(row[4])
                # Check if time is above bound for at least one side
                time_above_bound = any(
                    [x > args.check_longer_than for x in [time_a, time_b]]
                )
                # Compute B/A ratios (for % delta and geomean)
                time_ratio = time_b / time_a if time_a else float('nan')
                mem_ratio = mem_b / mem_a if mem_a else float('nan')
                # Keep ratios for geomean
                if time_above_bound and time_ratio > 0:  # must be >0 for gmean
                    time_ratios += [time_ratio]
                mem_ratios += [mem_ratio]
                # Deltas: (B/A)-1 = (B-A)/A
                time_delta = time_ratio - 1
                mem_delta = mem_ratio - 1
                # Check individual test results vs single test threshold
                time_exc = (
                    100.0 * abs(time_delta) > args.threshold_single and time_above_bound
                )
                mem_exc = 100.0 * abs(mem_delta) > args.threshold_single
                if time_exc or mem_exc:
                    threshold_single = True
                # Print deltas with formatting in verbose mode
                if args.verbose or time_exc or mem_exc:
                    print(
                        test_name,
                        fmt_delta(time_delta, time_exc, time_above_bound),
                        fmt_delta(mem_delta, mem_exc),
                    )

    time_gmean_delta = geometric_mean(time_ratios) - 1
    mem_gmean_delta = geometric_mean(mem_ratios) - 1
    time_agg_threshold = 100.0 * abs(time_gmean_delta) > args.threshold_agg
    mem_agg_threshold = 100.0 * abs(mem_gmean_delta) > args.threshold_agg
    if time_agg_threshold or mem_agg_threshold:
        threshold_agg = True
    if time_agg_threshold or mem_agg_threshold or args.verbose:
        print(
            "Geomean",
            fmt_delta(time_gmean_delta, time_agg_threshold),
            fmt_delta(mem_gmean_delta, mem_agg_threshold),
        )
    exit(threshold_single or threshold_agg)


if __name__ == "__main__":
    main()
