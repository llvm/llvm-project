// REQUIRES: arm
// RUN: llvm-mc -filetype=obj -triple=armv7a-none-linux-gnueabi %s -o %t
// RUN: llvm-mc -filetype=obj -triple=armv7a-none-linux-gnueabi %S/Inputs/abs256.s -o %t256.o
// RUN: ld.lld %t %t256.o -o %t2
// RUN: llvm-objdump -d %t2 | FileCheck %s
 .syntax unified
 .globl _start
_start:
 .section .R_ARM_ABS32POS, "ax",%progbits
 .word foo + 0x24

// S = 0x100, A = 0x24
// S + A = 0x124
// CHECK: Disassembly of section .R_ARM_ABS32POS:
// CHECK-EMPTY:
// CHECK: 00000124   
 .section .R_ARM_ABS32NEG, "ax",%progbits
 .word foo - 0x24
// S = 0x100, A = -0x24
// CHECK: Disassembly of section .R_ARM_ABS32NEG:
// CHECK-EMPTY:
// CHECK: 000000dc   
