# REQUIRES: riscv
# RUN: llvm-mc -filetype=obj -triple=riscv32 -mattr=+relax %s -o %t.32.o
# RUN: ld.lld -pie %t.32.o -o %t.32
# RUN: llvm-readelf -x .rodata -x .debug_info %t.32 | FileCheck --check-prefix=HEX %s

# RUN: llvm-mc -filetype=obj -triple=riscv64 -mattr=+relax %s -o %t.64.o
# RUN: ld.lld -shared %t.64.o -o %t.64
# RUN: llvm-readelf -x .rodata -x .debug_info %t.64 | FileCheck --check-prefix=HEX %s

# HEX:      section '.rodata':
# HEX-NEXT: 0x{{[0-9a-f]+}} 04000000 00000000 04000000 040004
# HEX:      section '.debug_info':
# HEX-NEXT: 0x{{[0-9a-f]+}} 04000000 00000000

## R_RISCV_ADD* and R_RISCV_SUB* are link-time constants, otherwise they are
## not allowed in -pie/-shared mode. Place .L0 and .L1 in distinct sections so
## the assembler cannot fold the differences.

.section .text.f0,"ax",@progbits
.globl _start
_start:
.L0:
  ret

.section .text.f1,"ax",@progbits
.L1:
  ret

.rodata
.dword .L1 - .L0
.word .L1 - .L0
.half .L1 - .L0
.byte .L1 - .L0

## Debug section may use R_RISCV_ADD64/R_RISCV_SUB64 pairs to measure lengths
## of code ranges (e.g. DW_AT_high_pc). Check we allow R_RISCV_ADD*/R_RISCV_SUB*
## in such non-SHF_ALLOC sections in -pie/-shared mode.
.section .debug_info
.quad .L1 - .L0
