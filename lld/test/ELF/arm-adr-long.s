// REQUIRES: arm
// RUN: llvm-mc --triple=armv7a-none-eabi --arm-add-build-attributes -filetype=obj -o %t.o %s
// RUN: echo "SECTIONS { \
// RUN:                 .text.0 0x10000000 : AT(0x10000000) { *(.text.0) } \
// RUN:                 .text.1 0x80000000 : AT(0x80000000) { *(.text.1) } \
// RUN:                 .text.2 0xf0000010 : AT(0xf0000010) { *(.text.2) } \
// RUN:               } " > %t.script
// RUN: ld.lld --script %t.script %t.o -o %t
// RUN: llvm-objdump -d --no-show-raw-insn --triple=armv7a-none-eabi %t | FileCheck %s

/// Test the long range encoding of R_ARM_ALU_PC_G0. We can encode an 8-bit
/// immediate rotated right by an even 4-bit field.
 .section .text.0, "ax", %progbits
dat1:
 .word 0

 .section .text.1, "ax", %progbits
 .global _start
 .type _start, %function
_start:
/// adr.w r0, dat1
 .inst 0xe24f0008
 .reloc 0, R_ARM_ALU_PC_G0, dat1
/// adr.w r1, dat2
 .inst 0xe24f1008
 .reloc 4, R_ARM_ALU_PC_G0, dat2

 .section .text.2, "ax", %progbits
dat2:
 .word 0

// CHECK:      10000000 <dat1>:
// CHECK-NEXT: 10000000: andeq   r0, r0, r0

// CHECK:      80000000 <_start>:
/// 0x80000000 + 0x8 - 0x70000008 = 0x10000000
// CHECK-NEXT: 80000000: sub     r0, pc, #1879048200
/// 0x80000004 + 0x8 + 0x70000004 = 0xf0000010
// CHECK-NEXT: 80000004: add     r1, pc, #1879048196

// CHECK:      f0000010 <dat2>:
// CHECK-NEXT: f0000010: andeq   r0, r0, r0
