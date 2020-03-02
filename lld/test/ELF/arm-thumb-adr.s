// REQUIRES: arm
// RUN: llvm-mc --triple=thumbv6m-none-eabi --arm-add-build-attributes -filetype=obj -o %t.o %s
// RUN: ld.lld %t.o -o %t
// RUN: llvm-objdump -d --no-show-raw-insn %t --triple=thumbv6m-none-eabi | FileCheck %s

/// Test R_ARM_THM_PC8 as used in the adr pseudo instruction. Only positive
/// 4-byte aligned offsets are permitted.
 .section .text.01, "ax", %progbits
 .balign 4
 .global _start
 .thumb_func
_start:
 adr r0, target1
 adr r1, target2

 .section .text.02, "ax", %progbits
 .balign 4
 .global target1
 .type target1, %function
target1:
 nop
 bx lr
 .section .text.03, "ax", %progbits
 .balign 4
 .space 1016
 .type target2, %function
target2:
 nop
 bx lr

// CHECK: 000110b4 _start:
// CHECK-NEXT: 110b4: adr     r0, #0
// CHECK-NEXT: 110b6: adr     r1, #1020

// CHECK: 000110b8 target1:
// CHECK-NEXT: 110b8: nop
// CHECK-NEXT: 110ba: bx      lr

// CHECK: 000114b4 target2:
// CHECK-NEXT: 114b4: nop
// CHECK-NEXT: 114b6: bx      lr
