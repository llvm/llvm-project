// REQUIRES: arm
// RUN: llvm-mc --triple=armv7a-none-eabi --arm-add-build-attributes -filetype=obj -o %t.o %s
// RUN: not ld.lld -n %t.o -o %t 2>&1 | FileCheck %s
 .section .text.0, "ax", %progbits
 .thumb_func
 .balign 4
low:
  bx lr
  nop
  nop

 .section .text.1, "ax", %progbits
 .global _start
 .arm
_start:
// CHECK: {{.*}}.s.tmp.o:(.text.1+0x0): relocation R_ARM_LDR_PC_G0 out of range: 4096 is not in [0, 4095]
/// ldr r0, low - 4076
 .inst 0xe51f0ff4
 .reloc 0, R_ARM_LDR_PC_G0, low
// CHECK: {{.*}}.s.tmp.o:(.text.1+0x4): relocation R_ARM_LDR_PC_G0 out of range: 4096 is not in [0, 4095]
/// ldr r0, high + 4100
 .inst 0xe59f0ffc
 .reloc 4, R_ARM_LDR_PC_G0, high

 .section .text.2
 .thumb_func
 .balign 4
high:
 bx lr
