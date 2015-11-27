// REQUIRES: aarch64
// RUN: llvm-mc -filetype=obj -triple=aarch64-none-freebsd %s -o %t.o
// RUN: llvm-mc -filetype=obj -triple=aarch64-none-freebsd %S/Inputs/abs255.s -o %t255.o
// RUN: llvm-mc -filetype=obj -triple=aarch64-none-freebsd %S/Inputs/abs256.s -o %t256.o
// RUN: llvm-mc -filetype=obj -triple=aarch64-none-freebsd %S/Inputs/abs257.s -o %t257.o

.data
  .word foo + 0xfffffeff
  .word foo - 0x80000100

// RUN: ld.lld -shared %t.o %t256.o -o %t.so
// RUN: llvm-objdump -s -section=.data %t.so | FileCheck %s

// CHECK: Contents of section .data:
// 1090: S = 0x100, A = 0xfffffeff
//       S + A = 0xffffffff
// 1094: S = 0x100, A = -0x80000100
//       S + A = 0x80000000
// CHECK-NEXT: 1090 ffffffff 00000080

// RUN: not ld.lld -shared %t.o %t255.o -o %t.so
//   | FileCheck %s --check-prefix=OVERFLOW
// RUN: not ld.lld -shared %t.o %t257.o -o %t.so
//   | FileCheck %s --check-prefix=OVERFLOW
// OVERFLOW: Relocation R_AARCH64_ABS32 out of range
