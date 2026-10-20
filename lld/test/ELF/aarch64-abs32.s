// REQUIRES: aarch64
// RUN: rm -rf %t && mkdir %t && cd %t

// RUN: llvm-mc -filetype=obj -triple=aarch64 %s -o a.o
// RUN: llvm-mc -filetype=obj -triple=aarch64_be %s -o a.be.o

// RUN: ld.lld a.o --defsym foo=256 -o a
// RUN: llvm-objdump -s --section=.data a | FileCheck %s --check-prefixes=CHECK,LE
// RUN: ld.lld a.be.o --defsym foo=256 -o a.be
// RUN: llvm-objdump -s --section=.data a.be | FileCheck %s --check-prefixes=CHECK,BE

// CHECK: Contents of section .data:
// 220158: S = 0x100, A = 0xfffffeff
//         S + A = 0xffffffff
// 22015c: S = 0x100, A = -0x80000100
//         S + A = 0x80000000
// LE-NEXT: 220158 ffffffff 00000080
// BE-NEXT: 220158 ffffffff 80000000

// RUN: not ld.lld a.o --defsym foo=255 2>&1 | FileCheck %s --check-prefix=OVERFLOW1 --implicit-check-not=error:
// OVERFLOW1: error: a.o:(.data+0x4): relocation R_AARCH64_ABS32 out of range: -2147483649 is not in [-2147483648, 4294967295]; references 'foo'

// RUN: not ld.lld a.o --defsym foo=257 2>&1 | FileCheck %s --check-prefix=OVERFLOW2 --implicit-check-not=error:
// OVERFLOW2: error: a.o:(.data+0x0): relocation R_AARCH64_ABS32 out of range: 4294967296 is not in [-2147483648, 4294967295]; references 'foo'

.globl _start
_start:
.data
  .word foo + 0xfffffeff
  .word foo - 0x80000100
