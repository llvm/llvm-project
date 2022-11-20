// REQUIRES: arm
// RUN: llvm-mc --triple=armv7a-none-eabi --arm-add-build-attributes -filetype=obj -o %t.o %s
// RUN: echo "SECTIONS { \
// RUN:                 .rodata.low 0x8012  : { *(.rodata.low) } \
// RUN:                 .text.low   0x8f00  : { *(.text.low) } \
// RUN:                 .text.neg   0x9000  : { *(.text.neg) } \
// RUN:                 .text.pos   0x10000 : { *(.text.pos) } \
// RUN:                 .text.high  0x10100 : { *(.text.high) } \
// RUN:                 .data_high  0x1100f : { *(.data.high) } \
// RUN:               } " > %t.script
// RUN: ld.lld -n --script %t.script %t.o -o %t
// RUN: llvm-readobj --symbols %t | FileCheck %s --check-prefix=SYMS
// RUN: llvm-objdump --no-print-imm-hex -d --no-show-raw-insn --triple=armv7a-none-eabi %t | FileCheck %s

/// Test the various legal cases for the R_ARM_LDR_PC_G0 relocation
/// Range is +- 4095 bytes
/// The Thumb bit for function symbols is ignored
 .section .rodata.low, "a", %progbits
dat1:
 .byte 0
dat2:
 .byte 1
dat3:
 .byte 2
dat4:
 .byte 3

 .section .text.low, "ax", %progbits
 .balign 4
 .global target1
 .type target1, %function
target1:
 bx lr
 .type target2, %function
target2:
 bx lr

 .section .text.neg, "ax", %progbits
 .balign 4
 .global _start
 .type _start, %function
_start:
/// ldr r0, dat1
 .inst 0xe51f0008
 .reloc 0, R_ARM_LDR_PC_G0, dat1
/// ldr r1, dat2
 .inst 0xe51f1008
 .reloc 4, R_ARM_LDR_PC_G0, dat2
/// ldr r2, dat3
 .inst 0xe51f2008
 .reloc 8, R_ARM_LDR_PC_G0, dat3
/// ldr r3, dat4
 .inst 0xe51f3008
 .reloc 0xc, R_ARM_LDR_PC_G0, dat4
/// ldr r0, target1
 .inst 0xe51f0008
 .reloc 0x10, R_ARM_LDR_PC_G0, target1
/// ldr r1, target2
 .inst 0xe51f1008
 .reloc 0x14, R_ARM_LDR_PC_G0, target2

 .section .text.pos, "ax", %progbits
 .balign 4
 .global pos
 .type pos, %function
pos:
/// ldr r2, target3
 .inst 0xe51f2008
 .reloc 0, R_ARM_LDR_PC_G0, target3
/// ldr r3, target4
 .inst 0xe51f3008
 .reloc 4, R_ARM_LDR_PC_G0, target4
/// ldr r0, dat5
 .inst 0xe51f0008
 .reloc 8, R_ARM_LDR_PC_G0, dat5
/// ldr r1, dat6
 .inst 0xe51f1008
 .reloc 0xc, R_ARM_LDR_PC_G0, dat6
/// ldr r2, dat7
 .inst 0xe51f2008
 .reloc 0x10, R_ARM_LDR_PC_G0, dat7
/// ldr r3, dat8
 .inst 0xe51f3008
 .reloc 0x14, R_ARM_LDR_PC_G0, dat8

/// positive addend in instruction, all others are -4 (PC bias)
///ldr r4, dat5 + 8
 .inst 0xe59f4000
 .reloc 0x18, R_ARM_LDR_PC_G0, dat5

 .section .text.high, "ax", %progbits
 .balign 4
 .type target3, %function
 .global target3
target3:
 bx lr
 .thumb_func
target4:
 bx lr

 .section .data.high, "aw", %progbits
dat5:
 .byte 0
dat6:
 .byte 1
dat7:
 .byte 2
dat8:
 .byte 3

// SYMS:     Name: dat1
// SYMS-NEXT:     Value: 0x8012
// SYMS:     Name: dat2
// SYMS-NEXT:     Value: 0x8013
// SYMS:     Name: dat3
// SYMS-NEXT:     Value: 0x8014
// SYMS:     Name: dat4
// SYMS-NEXT:     Value: 0x8015

// CHECK: 00008f00 <target1>:
// CHECK-NEXT:     8f00:        bx      lr

// CHECK: 00008f04 <target2>:
// CHECK-NEXT:     8f04:        bx      lr

// CHECK: 00009000 <_start>:
/// 0x9000 + 0x8 - 0xff6 = 0x8012
// CHECK-NEXT: 9000:  ldr     r0, [pc, #-4086]
/// 0x9004 + 0x8 - 0xff9 = 0x8013
// CHECK-NEXT: 9004:  ldr     r1, [pc, #-4089]
/// 0x9008 + 0x8 - 0xffc = 0x8014
// CHECK-NEXT: 9008:  ldr     r2, [pc, #-4092]
/// 0x900c + 0x8 - 0xfff = 0x8015
// CHECK-NEXT: 900c:  ldr     r3, [pc, #-4095]
/// 0x9010 + 0x8 - 0x118 = 0x8f00
// CHECK-NEXT: 9010:  ldr     r0, [pc, #-280]
/// 0x9014 + 0x8 - 0x118 = 0x8f04
// CHECK-NEXT: 9014:  ldr     r1, [pc, #-280]
///
// CHECK: 00010000 <pos>:
/// 0x10000 + 0x8 + 0xf8 = 0x10100
// CHECK-NEXT: 10000:  ldr     r2, [pc, #248]
/// 0x10004 + 0x8 + 0xf8 = 0x10104
// CHECK-NEXT: 10004: ldr     r3, [pc, #248]
/// 0x10008 + 0x8 + 0xfff = 0x1100f
// CHECK-NEXT: 10008: ldr     r0, [pc, #4095]
/// 0x1000c + 0x8 + 0xffc = 0x11010
// CHECK-NEXT: 1000c: ldr     r1, [pc, #4092]
/// 0x10010 + 0x8 + 0xff9 = 0x11011
// CHECK-NEXT: 10010: ldr     r2, [pc, #4089]
/// 0x10014 + 0x8 + 0xff6 = 0x11012
// CHECK-NEXT: 10014: ldr     r3, [pc, #4086]
/// 0x10018 + 0x8 + 0xff7 = 0x11017 = dat5 + 8
// CHECK-NEXT: 10018: ldr     r4, [pc, #4087]

// CHECK: 00010100 <target3>:
// CHECK-NEXT: 10100: bx      lr

// CHECK: 00010104 <target4>:
// CHECK-NEXT: 10104: bx      lr

// SYMS:     Name: dat5
// SYMS-NEXT:     Value: 0x1100F
// SYMS:     Name: dat6
// SYMS-NEXT:     Value: 0x11010
// SYMS:     Name: dat7
// SYMS-NEXT:     Value: 0x11011
// SYMS:     Name: dat8
// SYMS-NEXT:     Value: 0x11012
