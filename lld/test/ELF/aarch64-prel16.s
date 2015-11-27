// REQUIRES: aarch64
// RUN: llvm-mc -filetype=obj -triple=aarch64-none-freebsd %s -o %t.o
// RUN: llvm-mc -filetype=obj -triple=aarch64-none-freebsd %S/Inputs/abs255.s -o %t255.o
// RUN: llvm-mc -filetype=obj -triple=aarch64-none-freebsd %S/Inputs/abs256.s -o %t256.o
// RUN: llvm-mc -filetype=obj -triple=aarch64-none-freebsd %S/Inputs/abs257.s -o %t257.o

.data
  .hword foo - . + 0x10f8f
  .hword foo - . - 0x706e

// Note: If this test fails, it is probably results from
//       the change of the address of the .data section.
//       You may found the correct address in the aarch64_abs16.s test,
//       if it's already fixed. Then, update addends accordingly.
// RUN: ld.lld -shared %t.o %t256.o -o %t.so
// RUN: llvm-objdump -s -section=.data %t.so | FileCheck %s

// CHECK: Contents of section .data:
// 1090: S = 0x100, A = 0x10f8f, P = 0x1090
//       S + A - P = 0xffff
// 1092: S = 0x100, A = -0x706e, P = 0x1092
//       S + A - P = 0x8000
// CHECK-NEXT: 1090 ffff0080

// RUN: not ld.lld -shared %t.o %t255.o -o %t.so
//   | FileCheck %s --check-prefix=OVERFLOW
// RUN: not ld.lld -shared %t.o %t257.o -o %t.so
//   | FileCheck %s --check-prefix=OVERFLOW
// OVERFLOW: Relocation R_AARCH64_PREL16 out of range
