// REQUIRES: arm
// RUN: llvm-mc --triple=thumbv6m-none-eabi --arm-add-build-attributes -filetype=obj -o %t.o %s
// RUN: not ld.lld %t.o -o /dev/null 2>&1 | FileCheck %s

 .section .text.0, "ax", %progbits
 .balign 4
 .thumb_func
low:
 bx lr

 .section .text.1, "ax", %progbits
 .balign 2
 .global _start
 .thumb_func
_start:
// CHECK: {{.*}}.s.tmp.o:(.text.1+0x0): relocation R_ARM_THM_PC8 out of range: 18446744073709551612 is not in [0, 1023]
/// ldr r0, low
 .inst.n 0x48ff
 .reloc 0, R_ARM_THM_PC8, low
// CHECK: {{.*}}.s.tmp.o:(.text.1+0x2): improper alignment for relocation R_ARM_THM_PC8: 0x2 is not aligned to 4 bytes
/// ldr r1, unaligned
 .inst.n 0x49ff
 .reloc 2, R_ARM_THM_PC8, unaligned
// CHECK: {{.*}}.s.tmp.o:(.text.1+0x4): relocation R_ARM_THM_PC8 out of range: 1024 is not in [0, 1023]
/// ldr r2, range
 .inst.n 0x4aff
 .reloc 4, R_ARM_THM_PC8, range

 .section .text.2, "ax", %progbits
 .balign 4
 nop
 .thumb_func
unaligned:
  bx lr
 .space 1020
range:
  bx lr
