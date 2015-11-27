// RUN: llvm-mc -filetype=obj -triple=aarch64-none-freebsd %s -o %t
// RUN: ld.lld -shared %t -o %t2
// RUN: llvm-objdump -s %t2 | FileCheck %s
// REQUIRES: aarch64

.section .R_AARCH64_ABS64, "ax",@progbits
  .xword sym + 36

// CHECK: Contents of section .R_AARCH64_ABS64:
// CHECK-NEXT: 1000 24000000 00000000
//                  ^-- A = 0x24

.section .R_AARCH64_PREL64, "ax",@progbits
  .xword sym - . + 36

// S + A = 0x24
// P = 0x1008
// SA - P = 0xfffffffffffff01c
// CHECK: Contents of section .R_AARCH64_PREL64:
// CHECK-NEXT: 1008 1cf0ffff ffffffff
