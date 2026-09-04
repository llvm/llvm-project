#!/usr/bin/env python3
"""Generate payload-dump kernel variants from the ocloc-disassembled reference.

Each variant stores a GRF pair (rX, rX+1) per lane into out[gid], letting the
host read the raw thread payload as the hardware delivered it.
"""
import re
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "out" / "probe"
REF_DUMP = HERE.parent / "out" / "mod"  # stage-1 disasm of the ref kernel

VARIANTS = [0, 2, 4, 6]  # dump r0/r1, r2/r3, r4/r5, r6/r7

asm = (REF_DUMP / ".text.scale.asm").read_text().splitlines(keepends=True)

# Replace the data-producing add (gid*4 + bias -> r13) with a register dump
# read. The store data payload is r13:2 (two GRFs, one dword per lane).
data_add = re.compile(r"\s*add \(32\|M0\)\s+r13\.0<1>:d\s+r11\.0.*")
n_replaced = 0
for i, line in enumerate(asm):
    if data_add.match(line):
        n_replaced += 1
        asm[i] = "DUMP_MARK\n"
assert n_replaced == 1, f"expected 1 data add, found {n_replaced}"

# Fix the store's dependency: the dump mov immediately precedes it now.
store_re = re.compile(r"(send\.ugm \(32\|M0\).*)")

for variant in VARIANTS:
    lines = []
    for line in asm:
        if line == "DUMP_MARK\n":
            # Wait for all payload loads, then read the GRF pair per lane.
            lines.append("        sync.allrd                         null\n")
            lines.append(
                f"        mov (32|M0)              r13.0<1>:ud   r{variant}.0<1;1,0>:ud\n"
            )
            continue
        m = store_re.match(line)
        if m:
            # Store depends on the dump mov above it.
            line = re.sub(r"\{[^}]*\}", "{@1,$3}", line)
        lines.append(line)
    work = OUT / f"dump_r{variant}"
    work.mkdir(parents=True, exist_ok=True)
    for f in REF_DUMP.iterdir():
        if f.is_file():
            shutil.copy(f, work / f.name)
    (work / ".text.scale.asm").write_text("".join(lines))
    subprocess.run(
        ["ocloc", "asm", "-file", f"dump_r{variant}.bin", "-dump", str(work),
         "-device", "bmg-g21"],
        cwd=OUT, check=True, capture_output=True)
    print(f"built {work / f'dump_r{variant}.bin'}")

print("done")
