// REQUIRES: arm
// RUN: split-file %s %t
// RUN: llvm-mc --triple=armv7a-none-eabi --arm-add-build-attributes -filetype=obj -o %t.o %t/asm
// RUN: ld.lld --script %t/lds %t.o -o %t2
// RUN: llvm-objdump -d --no-show-raw-insn --triple=armv7a-none-eabi %t2 | FileCheck %s

/// Test the long range encoding of R_ARM_ALU_PC_Gx sequences. We can encode an 8-bit
/// immediate rotated right by an even 4-bit field.

//--- lds
SECTIONS {
    .text.0 0x0100000 : AT(0x0100000) { *(.text.0) }
    .text.1 0x0800000 : AT(0x0800000) { *(.text.1) }
    .text.2 0xf0f0000 : AT(0xf0f0000) { *(.text.2) }
}

//--- asm
 .section .text.0, "ax", %progbits
dat1:
 .word 0

 .section .text.1, "ax", %progbits
 .global _start
 .type _start, %function
_start:
 .inst 0xe24f0008 // sub r0, pc, #8
 .inst 0xe2400004 // sub r0, r0, #4
 .reloc 0, R_ARM_ALU_PC_G0_NC, dat1
 .reloc 4, R_ARM_ALU_PC_G1, dat1

 .inst 0xe24f1008 // sub r1, pc, #8
 .inst 0xe2411004 // sub r1, r1, #4
 .inst 0xe2411000 // sub r1, r1, #0
 .reloc 8, R_ARM_ALU_PC_G0_NC, dat2
 .reloc 12, R_ARM_ALU_PC_G1_NC, dat2
 .reloc 16, R_ARM_ALU_PC_G2, dat2

 .inst 0xe24f0008 // sub r0, pc, #8
 .inst 0xe2400004 // sub r0, r0, #4
 .inst 0xe2400000 // sub r0, r0, #0
 .reloc 20, R_ARM_ALU_PC_G0_NC, dat1
 .reloc 24, R_ARM_ALU_PC_G1_NC, dat1
 .reloc 28, R_ARM_ALU_PC_G2, dat1

 .inst 0xe24f0008 // sub r0, pc, #8
 .inst 0xe2400004 // sub r0, r0, #4
 .inst 0xe5900000 // ldr r0, [r0, #0]
 .reloc 32, R_ARM_ALU_PC_G0_NC, dat2
 .reloc 36, R_ARM_ALU_PC_G1_NC, dat2
 .reloc 40, R_ARM_LDR_PC_G2, dat2

 .inst 0xe24f0008 // sub r0, pc, #8
 .inst 0xe5100004 // ldr r0, [r0, #-4]
 .reloc 44, R_ARM_ALU_PC_G0_NC, dat1
 .reloc 48, R_ARM_LDR_PC_G1, dat1

 .inst 0xe24f0008 // sub r0, pc, #8
 .inst 0xe2400004 // sub r0, r0, #4
 .inst 0xe5900000 // ldr r0, [r0, #0]
 .reloc 52, R_ARM_ALU_PC_G0_NC, dat1
 .reloc 56, R_ARM_ALU_PC_G1_NC, dat1
 .reloc 60, R_ARM_LDR_PC_G2, dat1

 .inst 0xe24f0008 // sub r0, pc, #8
 .inst 0xe14000d4 // ldrd r0, [r0, #-4]
 .reloc 64, R_ARM_ALU_PC_G0_NC, dat1
 .reloc 68, R_ARM_LDRS_PC_G1, dat1

 .section .text.2, "ax", %progbits
dat2:
 .word 0

// CHECK:      00100000 <dat1>:
// CHECK-NEXT:   100000: andeq   r0, r0, r0

// CHECK:      00800000 <_start>:
// CHECK-NEXT:   800000: sub     r0, pc, #112, #16
// CHECK-NEXT:   800004: sub     r0, r0, #8

// CHECK-NEXT:   800008: add     r1, pc, #232, #12
// CHECK-NEXT:   80000c: add     r1, r1, #978944
// CHECK-NEXT:   800010: add     r1, r1, #4080

// CHECK-NEXT:   800014: sub     r0, pc, #112, #16
// CHECK-NEXT:   800018: sub     r0, r0, #28
// CHECK-NEXT:   80001c: sub     r0, r0, #0

// CHECK-NEXT:   800020: add     r0, pc, #232, #12
// CHECK-NEXT:   800024: add     r0, r0, #978944
// CHECK-NEXT:   800028: ldr     r0, [r0, #4056]

// CHECK-NEXT:   80002c: sub     r0, pc, #112, #16
// CHECK-NEXT:   800030: ldr     r0, [r0, #-52]

// CHECK-NEXT:   800034: sub     r0, pc, #112, #16
// CHECK-NEXT:   800038: sub     r0, r0, #60
// CHECK-NEXT:   80003c: ldr     r0, [r0, #-0]

// CHECK-NEXT:   800040: sub     r0, pc, #112, #16
// CHECK-NEXT:   800044: ldrd    r0, r1, [r0, #-72]

// CHECK:      0f0f0000 <dat2>:
// CHECK-NEXT:  f0f0000: andeq   r0, r0, r0
