# REQUIRES: x86

# RUN: split-file %s %t.dir && cd %t.dir
# RUN: llvm-mc -filetype=obj -triple=x86_64-windows input.s -o input.obj
# RUN: lld-link /dll /entry:main /nodefaultlib /threads:1 /debug /debugtype:cv /out:plain.dll input.obj
# RUN: lld-link /dll /entry:main /nodefaultlib /threads:1 /debug /debugtype:cv,fixup /out:fixup.dll input.obj
# RUN: llvm-objcopy --dump-section .pdata=plain.pdata plain.dll plain-copy.dll
# RUN: llvm-objcopy --dump-section .pdata=fixup.pdata fixup.dll fixup-copy.dll
# RUN: diff plain.pdata fixup.pdata
# RUN: llvm-readobj --file-headers fixup.dll > headers.txt
# RUN: llvm-pdbutil dump --fixups fixup.pdb > fixups.txt
# RUN: %python check.py fixup.pdata headers.txt fixups.txt

#--- input.s
        .text
        .globl main
main:
        retq
        .fill 7, 1, 0x90

        .macro runtime_function
        .section .pdata,"dr"
        .rva main + ((i * 17) % 7)
        .rva main + ((i * 17) % 7) + 1
        .rva unwind\@
        .section .rdata,"dr"
        .globl unwind\@
unwind\@:
        .long 1
        .endm

# Interleave tied start RVAs. Give each entry a distinct unwind target so
# that the test can match it to its own FIXUP regardless of tie order.
        .set i, 0
        .rept 128
        runtime_function
        .set i, i + 1
        .endr

#--- check.py
import pathlib
import re
import struct
import sys

pdata, headers, fixups = map(pathlib.Path, sys.argv[1:])
headers = headers.read_text()
base = int(re.search(r"ExceptionTableRVA: (0x[0-9A-Fa-f]+)", headers)[1], 16)
size = int(re.search(r"ExceptionTableSize: (0x[0-9A-Fa-f]+)", headers)[1], 16)
entries = struct.iter_unpack("<III", pdata.read_bytes()[:size])
expected = [(base + 12 * i + 8, target) for i, (_, _, target) in enumerate(entries)]
actual = [
    (int(site, 16), int(target, 16))
    for site, target in re.findall(
        r"^ *([0-9A-F]+) \| \.pdata +\| ([0-9A-F]+) +\| \.rdata +\| +3 \|",
        fixups.read_text(),
        re.M,
    )
]
assert actual == expected, "Unwind FIXUPs do not match the sorted .pdata entries"
