// REQUIRES: arm
// RUN: llvm-mc -filetype=obj -triple=armv7a-none-linux-gnueabi %s -o %t.o
// RUN: llvm-mc -filetype=obj -triple=armv7a-none-linux-gnueabi %S/Inputs/abs256.s -o %t256.o
// RUN: ld.lld %t.o %t256.o -o %t
// RUN: llvm-objdump -s %t | FileCheck %s --check-prefixes=CHECK,LE

// RUN: llvm-mc -filetype=obj -triple=armv7aeb-none-linux-gnueabi %s -o %t.be.o
// RUN: llvm-mc -filetype=obj -triple=armv7aeb-none-linux-gnueabi %S/Inputs/abs256.s -o %t256.be.o
// RUN: ld.lld %t.be.o %t256.be.o -o %t.be
// RUN: llvm-objdump -s %t.be | FileCheck %s --check-prefixes=CHECK,BE

// RUN: ld.lld --be8 %t.be.o %t256.be.o -o %t.be8
// RUN: llvm-objdump -s %t.be8 | FileCheck %s --check-prefixes=CHECK,BE

.globl _start
_start:
.section .R_ARM_ABS, "ax","progbits"
  .word foo + 0x24

// CHECK: Contents of section .R_ARM_ABS:
// LE-NEXT:  200b4 24010000
// BE-NEXT:  200b4 00000124

.section .R_ARM_PREL, "ax","progbits"
  .word foo - . + 0x24

// CHECK: Contents of section .R_ARM_PREL:
// LE-NEXT:  200b8 6c00feff
// BE-NEXT:  200b8 fffe006c

