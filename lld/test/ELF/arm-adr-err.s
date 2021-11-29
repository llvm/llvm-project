// REQUIRES: arm
// RUN: llvm-mc --triple=armv7a-none-eabi --arm-add-build-attributes -filetype=obj -o %t.o %s
// RUN: not ld.lld %t.o -o /dev/null 2>&1 | FileCheck %s
 .section .os0, "ax", %progbits
 .balign 1024
 .thumb_func
low:
 bx lr

/// Check that we error when the immediate for the add or sub is not encodeable
 .section .os1, "ax", %progbits
 .arm
 .balign 1024
 .global _start
 .type _start, %function
_start:
// CHECK: {{.*}}.s.tmp.o:(.os1+0x0): unencodeable immediate 1031 for relocation R_ARM_ALU_PC_G0
/// adr r0, low
 .inst 0xe24f0008
 .reloc 0, R_ARM_ALU_PC_G0, low
 // CHECK: {{.*}}.s.tmp.o:(.os1+0x4): unencodeable immediate 1013 for relocation R_ARM_ALU_PC_G0
/// adr r1, unaligned
 .inst 0xe24f1008
 .reloc 4, R_ARM_ALU_PC_G0, unaligned

 .balign 512
/// ldrd r0, r1, _start
// CHECK: {{.*}}.s.tmp.o:(.os1+0x200): relocation R_ARM_LDRS_PC_G0 out of range: 512 is not in [0, 255]; references _start
 .reloc ., R_ARM_LDRS_PC_G0, _start
 .inst 0xe14f00d0

 .section .os2, "ax", %progbits
 .balign 1024
 .thumb_func
unaligned:
  bx lr
