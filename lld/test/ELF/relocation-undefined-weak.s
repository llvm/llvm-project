# REQUIRES: x86

## On some targets (e.g. ARM, AArch64, and PPC), PC relative relocations to
## weak undefined symbols resolve to special positions. On many others
## the target symbols as treated as VA 0.

# RUN: llvm-mc -filetype=obj -triple=x86_64 %s -o %t.o
# RUN: ld.lld %t.o -o %t
# RUN: llvm-objdump -d --no-show-raw-insn --print-imm-hex %t | FileCheck %s

.global _start
_start:
  movl $1, sym1(%rip)

.weak sym1

# CHECK: 201120: movl $0x1, -0x20112a(%rip)
