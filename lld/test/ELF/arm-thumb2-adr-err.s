// REQUIRES: arm
// RUN: llvm-mc --triple=thumbv7m-none-eabi --arm-add-build-attributes -filetype=obj -o %t.o %s
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
// CHECK: {{.*}}.s.tmp.o:(.text.1+0x0): relocation R_ARM_THM_ALU_PREL_11_0 out of range: 4098 is not in [0, 4095]
/// adr.w r0, low - 4091
 .inst.w 0xf6af70ff
 .reloc 0, R_ARM_THM_ALU_PREL_11_0, low
// CHECK: {{.*}}.s.tmp.o:(.text.1+0x4): relocation R_ARM_THM_ALU_PREL_11_0 out of range: 4096 is not in [0, 4095]
/// adr.w r0, high + 4091
 .inst.w 0xf60f70f7
 .reloc 4, R_ARM_THM_ALU_PREL_11_0, high
 .section .text.2
 .thumb_func
 .balign 4
high:
 bx lr
