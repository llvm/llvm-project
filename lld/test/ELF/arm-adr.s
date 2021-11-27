// REQUIRES: arm
// RUN: llvm-mc --triple=armv7a-none-eabi --arm-add-build-attributes -filetype=obj -o %t.o %s
// RUN: ld.lld %t.o -o %t
// RUN: llvm-objdump -d --no-show-raw-insn --triple=armv7a-none-eabi %t | FileCheck %s

/// Test the short range cases of R_ARM_ALU_PC_G0. The range of the instruction
/// depends on the number of trailing zeros of the displacement. In practice
/// the maximum effective range will be 1024 bytes, which is a 4-byte aligned
/// instruction to a 4-byte aligned word.

 .arm
 .section .os1, "ax", %progbits
 .balign 1024
 .word 0
 .word 0
 .word 0
 .word 0
dat1:
 .word 0
dat2:
 .word 0

 .section .os2, "ax", %progbits
 .balign 1024
 .global _start
 .type _start, %function
_start:
/// adr r0, dat1
 .inst 0xe24f0008
 .reloc 0, R_ARM_ALU_PC_G0, dat1
/// adr r0, dat2
 .inst 0xe24f0008
 .reloc 4, R_ARM_ALU_PC_G0, dat2
/// adr r0, dat3
 .inst 0xe24f0008
 .reloc 8, R_ARM_ALU_PC_G0, dat3
/// adr r0, dat4
 .inst 0xe24f0008
 .reloc 0xc, R_ARM_ALU_PC_G0, dat4

 .section .os3, "ax", %progbits
 .balign 1024
 .word 0
 .word 0
 .word 0
 .word 0
dat3:
 .word 0
dat4:
 .word 0

 .section .os4, "ax", %progbits
 .thumb
 .type tfunc, %function
tfunc:
  bx lr

 .section .os5, "ax", %progbits
 .arm
 .type arm_func, %function

arm_func:
 .balign 4
/// adr r0, tfunc
 .inst 0xe24f0008
 .reloc 0, R_ARM_ALU_PC_G0, tfunc
/// adr r0, afunc
 .inst 0xe24f0008
 .reloc 4, R_ARM_ALU_PC_G0, afunc
 bx lr

 .section .os6, "ax", %progbits
 .type afunc, %function
 .balign 4
afunc:
 bx lr

// CHECK:      00020410 <dat1>:
// CHECK-NEXT: 20410: andeq   r0, r0, r0

// CHECK:      00020414 <dat2>:
// CHECK-NEXT: 20414: andeq   r0, r0, r0

// CHECK:     00020800 <_start>:
/// 0x20800 + 0x8 - 0x3f8 = 0x11410 = dat1
// CHECK-NEXT: 20800: sub     r0, pc, #1016
/// 0x20804 + 0x8 - 0x3f8 = 0x11414 = dat2
// CHECK-NEXT: 20804: sub     r0, pc, #1016
/// 0x20808 + 0x8 + 0x400 = 0x11c10 = dat3
// CHECK-NEXT: 20808: add     r0, pc, #64, #28
/// 0x2080c + 0x8 + 0x400 = 0x11c14 = dat4
// CHECK-NEXT: 2080c: add     r0, pc, #64, #28

// CHECK:      00020c10 <dat3>:
// CHECK-NEXT: 20c10: andeq   r0, r0, r0

// CHECK:      00020c14 <dat4>:
// CHECK-NEXT: 20c14: andeq   r0, r0, r0

// CHECK:      00020c18 <tfunc>:
// CHECK-NEXT: 20c18: bx      lr

// CHECK:      00020c1c <arm_func>:
/// 0x20c1c + 0x8 - 0xb = 11c19 = tfunc
// CHECK-NEXT: 20c1c: sub     r0, pc, #11
/// 0x20c20 + 0x8 = 0x11c28 = afunc
// CHECK-NEXT: 20c20: add     r0, pc, #0
// CHECK-NEXT: 20c24: bx      lr

// CHECK:      00020c28 <afunc>:
// CHECK-NEXT: 20c28: bx      lr
