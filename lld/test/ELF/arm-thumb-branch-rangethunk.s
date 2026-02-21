// REQUIRES: arm
// RUN: llvm-mc -arm-add-build-attributes -filetype=obj -triple=thumbv7a-none-linux-gnueabi %s -o %t
// RUN: llvm-mc -arm-add-build-attributes -filetype=obj -triple=thumbv7a-none-linux-gnueabi %S/Inputs/far-arm-thumb-abs.s -o %tfar
// RUN: ld.lld -Ttext=0x20000 %t %tfar -o %t2
// RUN: llvm-objdump --no-show-raw-insn -d %t2 | FileCheck %s
 .syntax unified
 .thumb
 .section .text, "ax",%progbits
 .globl _start
 .balign 0x10000
 .type _start,%function
_start:
 // address of too_far symbols are just out of range of ARM branch with
 // 26-bit immediate field and an addend of -8
 bl  too_far1
 b   too_far2
 beq.w too_far3

// CHECK-LABEL: 00020000 <_start>:
// CHECK-NEXT: 20000: bl      0x2000c <__Thumbv7ABSLongThunk_too_far1>
// CHECK-NEXT:        b.w     0x20010 <__Thumbv7ABSLongThunk_too_far2>
// CHECK-NEXT:        beq.w   0x20014 <__Thumbv7ABSLongThunk_too_far3>
// CHECK-EMPTY:
// CHECK-NEXT: 0002000c <__Thumbv7ABSLongThunk_too_far1>:
// CHECK-NEXT: 2000c: b.w     0x1020004
// CHECK-EMPTY:
// CHECK-NEXT: 00020010 <__Thumbv7ABSLongThunk_too_far2>:
// CHECK-NEXT: 20010: b.w     0x1020008
// CHECK-EMPTY:
// CHECK-NEXT: 00020014 <__Thumbv7ABSLongThunk_too_far3>:
// CHECK-NEXT: 20014: b.w     0x12000c
