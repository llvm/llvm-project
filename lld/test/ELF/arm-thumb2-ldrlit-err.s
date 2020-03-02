// REQUIRES: arm
// RUN: llvm-mc -g --triple=thumbv7m-none-eabi --arm-add-build-attributes -filetype=obj -o %t.o %s
// RUN: not ld.lld %t.o -o /dev/null 2>&1 | FileCheck %s

 .section .text.0, "ax", %progbits
 .thumb_func
 .balign 4
low:
  bx lr
  nop
  nop

 .section .text.1, "ax", %progbits
 .global _start
 .thumb_func
_start:
// CHECK: {{.*}}.s:[[# @LINE+1]]:(.text.1+0x0): relocation R_ARM_THM_PC12 out of range: 4098 is not in [0, 4095]
 ldr.w r0, low - 4091
// CHECK: {{.*}}.s:[[# @LINE+1]]:(.text.1+0x4): relocation R_ARM_THM_PC12 out of range: 4096 is not in [0, 4095]
 ldr.w r0, high + 4091

 .section .text.2
 .thumb_func
 .balign 4
high:
 bx lr
