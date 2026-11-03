#!/usr/bin/env python3
"""Verify launcher dump output against an expected expression.

Usage: launcher ... | verify.py '<expr>'   # expr over i for out0, e.g. 'i*2+7'
"""
import re
import sys

if sys.argv[1] == "--sort":
    # Permutation check: out values must be exactly 0..n-1 in some order.
    got = []
    for line in sys.stdin:
        m = re.match(r"out0\[(\d+)\] = 0x([0-9a-f]+)", line.strip())
        if m:
            got.append(int(m.group(2), 16))
    if sorted(got) == list(range(len(got))) and got:
        print(f"PASS: {len(got)} lanes, permutation of 0..{len(got)-1}")
        sys.exit(0)
    print("FAIL: not a permutation of 0..n-1", file=sys.stderr)
    sys.exit(1)

expr = sys.argv[1]
code = compile(expr, "<expect>", "eval")
bad = 0
total = 0
for line in sys.stdin:
    m = re.match(r"out0\[(\d+)\] = 0x([0-9a-f]+)", line.strip())
    if not m:
        continue
    i, got = int(m.group(1)), int(m.group(2), 16)
    want = eval(code, {"i": i}) & 0xFFFFFFFF
    total += 1
    if got != want:
        if bad < 8:
            print(f"  out[{i}] = 0x{got:08x}, want 0x{want:08x}", file=sys.stderr)
        bad += 1
if bad or total == 0:
    print(f"FAIL: {bad}/{total} mismatches", file=sys.stderr)
    sys.exit(1)
print(f"PASS: {total} lanes, out[i] == {expr}")
