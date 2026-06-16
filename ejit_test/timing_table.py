#!/usr/bin/env python3
"""Run ejit_timing_test N times and print a formatted timing summary table.

Usage:
    python3 timing_table.py [binary] [-n RUNS] [--ci CI] [--ti TI]

Default binary: ./out/ejit_timing_test
"""

import argparse
import os
import re
import subprocess
import sys


def parse_run(output):
    """Parse one run of ejit_timing_test output. Returns dict of metric→ns (int)."""
    m = {}

    # [1] Baseline
    hit = re.search(r'cell_nojit\s+avg=\s*(\d+)\s*ns/call', output)
    if hit: m['baseline'] = int(hit.group(1))

    # [2] Fallback
    hit = re.search(r'cell_jit \(fallback\)\s+avg=\s*(\d+)\s*ns/call', output)
    if hit: m['fallback'] = int(hit.group(1))

    # [3] 1st JIT compile – 1-dim (cell_jit, labelled with "JIT compile")
    hit = re.search(r'first call:\s+(\d+)\s*ns\s+\(JIT compile', output)
    if hit: m['compile_1dim'] = int(hit.group(1))

    # [4] Cache hit
    hit = re.search(r'cell_jit \(cache hit\)\s+avg=\s*(\d+)\s*ns/call', output)
    if hit: m['cache_hit'] = int(hit.group(1))

    # [5] 1st JIT compile – 2-dim (multi_jit, first "first call (cold):")
    # [6] 1st JIT compile – 0-dim (simple_jit, second "first call (cold):")
    cold = re.findall(r'first call \(cold\):\s+(\d+)\s*ns', output)
    if len(cold) >= 1: m['compile_2dim'] = int(cold[0])
    if len(cold) >= 2: m['compile_0dim'] = int(cold[1])

    # [7] Deactivate (single-shot, not a batch average)
    hit = re.search(r'deactivate \(w/ cache\):\s+(\d+)\s*ns', output)
    if hit: m['deactivate'] = int(hit.group(1))

    # [7] Activate (already-active batch average)
    hit = re.search(r'activate \(already active\): avg=(\d+)\s*ns/call', output)
    if hit: m['activate'] = int(hit.group(1))

    return m


def fmt_value(ns):
    """Format an integer nanosecond value with the appropriate SI unit."""
    if ns >= 1_000_000:
        return f'{ns / 1_000_000:.2f}ms'
    if ns >= 1_000:
        return f'{ns / 1_000:.2f}μs'   # µs
    return f'{ns}ns'


def fmt_jitter(values):
    """Return ±X.X% (half-range / mean) or '0%' when negligible."""
    if len(values) < 2:
        return '0%'
    mean = sum(values) / len(values)
    if mean == 0:
        return '0%'
    half_range = (max(values) - min(values)) / 2
    pct = half_range / mean * 100
    if pct < 0.5:
        return '0%'
    return f'±{pct:.1f}%'   # ±


ROWS = [
    ('baseline',     'Baseline (no EJIT)'),
    ('fallback',     'Fallback (wrapper, not active)'),
    ('cache_hit',    'Cache hit'),
    ('compile_1dim', '1st JIT compile (1-dim)'),
    ('compile_2dim', '1st JIT compile (2-dim)'),
    ('compile_0dim', '1st JIT compile (0-dim)'),
    ('deactivate',   'Deactivate (w/ invalidation)'),
    ('activate',     'Activate (already-active)'),
]

COL_METRIC = 36
COL_MEAN   = 12


def build_binary(here, binary):
    """Invoke build.sh to compile ejit_timing_test.  Uses lipo/ejit.o when present."""
    build_sh = os.path.join(here, 'build.sh')
    if not os.path.isfile(build_sh):
        sys.exit(f'error: binary not found and build.sh missing: {build_sh}')

    cmd = ['bash', build_sh]
    lipo = os.path.join(here, 'lipo', 'ejit.o')
    if os.path.isfile(lipo):
        cmd += [f'--lipo={lipo}']
    cmd += ['ejit_timing_test']

    print('  building ejit_timing_test…', file=sys.stderr)
    result = subprocess.run(cmd, cwd=here)
    if result.returncode != 0:
        sys.exit('error: build failed')
    if not os.path.isfile(binary):
        sys.exit(f'error: build succeeded but binary still missing: {binary}')


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    default_bin = os.path.join(here, 'out', 'ejit_timing_test')

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('binary', nargs='?', default=default_bin,
                    help='ejit_timing_test binary (default: out/ejit_timing_test)')
    ap.add_argument('-n', '--runs', type=int, default=5,
                    help='number of runs (default: 5)')
    ap.add_argument('--ci', type=int, default=0, help='cellIdx (default: 0)')
    ap.add_argument('--ti', type=int, default=0, help='trpIdx  (default: 0)')
    args = ap.parse_args()

    if not os.path.isfile(args.binary):
        build_binary(here, args.binary)

    collected: dict[str, list[int]] = {}

    for i in range(args.runs):
        print(f'  run {i + 1}/{args.runs}…', file=sys.stderr, end='\r')
        proc = subprocess.run(
            [args.binary, str(args.ci), str(args.ti)],
            capture_output=True, text=True,
        )
        for key, val in parse_run(proc.stdout).items():
            collected.setdefault(key, []).append(val)

    print(' ' * 40, file=sys.stderr, end='\r')   # erase progress line

    header = f"{'Metric':<{COL_METRIC}}{'Mean':<{COL_MEAN}}Jitter"
    print(header)
    print('─' * (COL_METRIC + COL_MEAN + 10))

    for key, label in ROWS:
        vals = collected.get(key, [])
        if not vals:
            mean_s, jitter_s = 'n/a', 'n/a'
        else:
            mean_s   = fmt_value(round(sum(vals) / len(vals)))
            jitter_s = fmt_jitter(vals)
        print(f'{label:<{COL_METRIC}}{mean_s:<{COL_MEAN}}{jitter_s}')


if __name__ == '__main__':
    main()