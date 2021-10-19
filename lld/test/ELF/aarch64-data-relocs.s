// REQUIRES: aarch64
// RUN: llvm-mc -filetype=obj -triple=aarch64 %s -o %t.o
// RUN: llvm-mc -filetype=obj -triple=aarch64 %S/Inputs/abs256.s -o %t256.o
// RUN: ld.lld %t.o %t256.o -o %t
// RUN: llvm-objdump -s %t | FileCheck %s --check-prefixes=CHECK,LE

// RUN: llvm-mc -filetype=obj -triple=aarch64_be %s -o %t.be.o
// RUN: llvm-mc -filetype=obj -triple=aarch64_be %S/Inputs/abs256.s -o %t256.be.o
// RUN: ld.lld %t.be.o %t256.be.o -o %t.be
// RUN: llvm-objdump -s %t.be | FileCheck %s --check-prefixes=CHECK,BE

.globl _start
_start:
.section .R_AARCH64_ABS64, "ax",@progbits
  .xword foo + 0x24

// S = 0x100, A = 0x24
// S + A = 0x124
// CHECK: Contents of section .R_AARCH64_ABS64:
// LE-NEXT: 210120 24010000 00000000
// BE-NEXT: 210120 00000000 00000124

.section .R_AARCH64_PREL64, "ax",@progbits
  .xword foo - . + 0x24

// S + A - P = 0x100 + 0x24 - 0x210128 = 0xffffffffffdefffc
// CHECK: Contents of section .R_AARCH64_PREL64:
// LE-NEXT: 210128 fcffdeff ffffffff
// BE-NEXT: 210128 ffffffff ffdefffc
