#!/usr/bin/env python3
#
# Determines which clang-tidy checks are "fast enough" to run in clangd.
# This runs clangd --check --check-tidy-time and parses the output.
# This program outputs a header fragment specifying which checks are fast:
#   FAST(bugprone-argument-comment, 5)
#   SLOW(misc-const-correctness, 200)
# If given the old header fragment as input, we lean to preserve its choices.
#
# This is not deterministic or hermetic, but should be run occasionally to
# update the list of allowed checks. From llvm-project:
#   clang-tools-extra/clangd/TidyFastChecks.py --clangd=build-opt/bin/clangd
# Be sure to use an optimized, no-asserts, tidy-enabled build of clangd!

import argparse
import os
import re
import subprocess
import sys

# Checks faster than FAST_THRESHOLD are fast, slower than SLOW_THRESHOLD slow.
# If a check is in between, we stick with our previous decision. This avoids
# enabling/disabling checks between releases due to random measurement jitter.
FAST_THRESHOLD = 8  # percent
SLOW_THRESHOLD = 15

parser = argparse.ArgumentParser()
parser.add_argument(
    "--target",
    help="X-macro output file. "
    "If it exists, existing contents will be used for hysteresis",
    default="clang-tools-extra/clangd/TidyFastChecks.inc",
)
parser.add_argument(
    "--source",
    help="Source file to benchmark tidy checks",
    default="clang/lib/Sema/Sema.cpp",
)
parser.add_argument(
    "--clangd", help="clangd binary to invoke", default="build/bin/clangd"
)
parser.add_argument("--checks", help="check glob to run", default="*")
parser.add_argument("--verbose", help="log clangd output", action="store_true")
args = parser.parse_args()

# Use the preprocessor to extract the list of previously-fast checks.
def read_old_fast(path):
    text = subprocess.check_output(
        [
            "cpp",
            "-P",  # Omit GNU line markers
            "-nostdinc",  # Don't include stdc-predef.h
            "-DFAST(C,T)=C",  # Print fast checks only
            path,
        ]
    )
    for line in text.splitlines():
        if line.strip():
            yield line.strip().decode("utf-8")


old_fast = list(read_old_fast(args.target)) if os.path.exists(args.target) else []
print(f"Old fast checks: {old_fast}", file=sys.stderr)

# Runs clangd --check --check-tidy-time.
# Yields (check, percent-overhead) pairs.
def measure():
    process = subprocess.Popen(
        [
            args.clangd,
            "--check=" + args.source,
            "--check-locations=0",  # Skip useless slow steps.
            "--check-tidy-time=" + args.checks,
        ],
        stderr=subprocess.PIPE,
    )
    recording = False
    for line in iter(process.stderr.readline, b""):
        if args.verbose:
            print("clangd> ", line, file=sys.stderr)
        if not recording:
            if b"Timing AST build with individual clang-tidy checks" in line:
                recording = True
            continue
        if b"Finished individual clang-tidy checks" in line:
            return
        match = re.search(rb"(\S+) = (\S+)%", line)
        if match:
            yield (match.group(1).decode("utf-8"), float(match.group(2)))


with open(args.target, "w", buffering=1) as target:
    # Produce an includable X-macros fragment with our decisions.
    print(
        f"""// This file is generated, do not edit it directly!
// Deltas are percentage regression in parsing {args.source}
#ifndef FAST
#define FAST(CHECK, DELTA)
#endif
#ifndef SLOW
#define SLOW(CHECK, DELTA)
#endif
""",
        file=target,
    )

    for check, time in measure():
        threshold = SLOW_THRESHOLD if check in old_fast else FAST_THRESHOLD
        decision = "FAST" if time <= threshold else "SLOW"
        print(f"{decision} {check} {time}% <= {threshold}%", file=sys.stderr)
        print(f"{decision}({check}, {time})", file=target)

    print(
        """
#undef FAST
#undef SLOW
""",
        file=target,
    )
