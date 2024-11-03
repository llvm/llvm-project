// REQUIRES: aarch64
// RUN: llvm-mc -filetype=obj -triple=aarch64 %s -o %t.o
// RUN: llvm-mc -filetype=obj -triple=aarch64 %S/Inputs/abs255.s -o %t255.o
// RUN: llvm-mc -filetype=obj -triple=aarch64 %S/Inputs/abs256.s -o %t256.o
// RUN: llvm-mc -filetype=obj -triple=aarch64 %S/Inputs/abs257.s -o %t257.o
// RUN: llvm-mc -filetype=obj -triple=aarch64_be %s -o %t.be.o
// RUN: llvm-mc -filetype=obj -triple=aarch64_be %S/Inputs/abs256.s -o %t256.be.o

/// Check for overflow with a R_AACH64_PLT32 relocation.

// RUN: ld.lld -z max-page-size=4096 %t.o %t256.o -o %t2
// RUN: llvm-objdump -s --section=.data %t2 | FileCheck %s --check-prefixes=CHECK,LE
// RUN: ld.lld -z max-page-size=4096 %t.be.o %t256.be.o -o %t2.be
// RUN: llvm-objdump -s --section=.data %t2.be | FileCheck %s --check-prefixes=CHECK,BE

// CHECK: Contents of section .data:
/// 202158: S = 0x100, A = 0x80202057, P = 0x202158
///         S + A - P = 0xffffff7f
/// 20215c: S = 0x100, A = -0x7fdfdfa4, P = 0x20215c
///         S + A - P = 0x80000000
/// 202160: S = 0x100, A = 0, P = 0x202160
///         S + A - P = 0xffdfdfa0
// LE-NEXT: 202158 ffffff7f 00000080 a0dfdfff
// BE-NEXT: 202158 7fffffff 80000000 ffdfdfa0

// RUN: not ld.lld -z max-page-size=4096 %t.o %t255.o -o /dev/null 2>&1 | FileCheck %s --check-prefix=OVERFLOW1
// OVERFLOW1: relocation R_AARCH64_PLT32 out of range: -2147483649 is not in [-2147483648, 2147483647]; references 'foo'

// RUN: not ld.lld -z max-page-size=4096 %t.o %t257.o -o /dev/null 2>&1 | FileCheck %s --check-prefix=OVERFLOW2
// OVERFLOW2: relocation R_AARCH64_PLT32 out of range: 2147483648 is not in [-2147483648, 2147483647]; references 'foo'

  .globl _start
  _start:
.data
  .word foo@PLT - . + 2149589079
  .word foo@PLT - . - 2145378212
  .word foo@PLT - .
