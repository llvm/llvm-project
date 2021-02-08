// REQUIRES: aarch64
// RUN: llvm-mc -filetype=obj -triple=aarch64 %s -o %t.o
// RUN: llvm-mc -filetype=obj -triple=aarch64 %S/Inputs/abs255.s -o %t255.o
// RUN: llvm-mc -filetype=obj -triple=aarch64 %S/Inputs/abs256.s -o %t256.o
// RUN: llvm-mc -filetype=obj -triple=aarch64 %S/Inputs/abs257.s -o %t257.o
// RUN: llvm-mc -filetype=obj -triple=aarch64_be %s -o %t.be.o
// RUN: llvm-mc -filetype=obj -triple=aarch64_be %S/Inputs/abs256.s -o %t256.be.o

.globl _start
_start:
.data
  .word foo - . + 0x100202057
  .word foo - . - 0x7fdfdfa4

// Note: If this test fails, it probably happens because of
//       the change of the address of the .data section.
//       You may found the correct address in the aarch64_abs32.s test,
//       if it is already fixed. Then, update addends accordingly.
// RUN: ld.lld -z max-page-size=4096 %t.o %t256.o -o %t
// RUN: llvm-objdump -s --section=.data %t | FileCheck %s --check-prefixes=CHECK,LE
// RUN: ld.lld -z max-page-size=4096 %t.be.o %t256.be.o -o %t.be
// RUN: llvm-objdump -s --section=.data %t.be | FileCheck %s --check-prefixes=CHECK,BE

// CHECK: Contents of section .data:
// 202158: S = 0x100, A = 0x100202057, P = 0x202158
//         S + A - P = 0xffffffff
// 20215c: S = 0x100, A = -0x7fdfdfa4, P = 0x20215c
//         S + A - P = 0x80000000
// LE-NEXT: 202158 ffffffff 00000080
// BE-NEXT: 202158 ffffffff 80000000

// RUN: not ld.lld -z max-page-size=4096 %t.o %t255.o -o /dev/null 2>&1 | FileCheck %s --check-prefix=OVERFLOW1
// OVERFLOW1: relocation R_AARCH64_PREL32 out of range: -2147483649 is not in [-2147483648, 4294967295]; references foo

// RUN: not ld.lld -z max-page-size=4096 %t.o %t257.o -o /dev/null 2>&1 | FileCheck %s --check-prefix=OVERFLOW2
// OVERFLOW2: relocation R_AARCH64_PREL32 out of range: 4294967296 is not in [-2147483648, 4294967295]; references foo
