// REQUIRES: arm
// RUN: llvm-mc --triple=thumbv7m-none-eabi --arm-add-build-attributes -filetype=obj -o %t.o %s
// RUN: echo "SECTIONS { \
// RUN:                 .rodata.low 0x8012  : { *(.rodata.low) } \
// RUN:                 .text.low   0x8f00  : { *(.text.low) } \
// RUN:                 .text.neg   0x9000  : { *(.text.neg) } \
// RUN:                 .text.pos   0x10000 : { *(.text.pos) } \
// RUN:                 .text.high  0x10100 : { *(.text.high) } \
// RUN:                 .data_high  0x1100f : { *(.data.high) } \
// RUN:               } " > %t.script
// RUN: ld.lld --script %t.script %t.o -o %t
// RUN: llvm-readobj --symbols %t | FileCheck %s --check-prefix=SYMS
// RUN: llvm-objdump -d --no-show-raw-insn %t | FileCheck %s

/// Test the various legal cases for the R_ARM_THM_ALU_PREL_11_0 relocation
/// Interesting things to note
/// Range is +- 4095 bytes
/// The expression is S + A - Pa where Pa is AlignDown(PC, 4) so we will use
/// 2-byte nops to make some of the adr psuedo instructions 2-byte aligned.
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
 .thumb_func
_start:
 nop
///  adr.w r0, dat1
 .inst.w 0xf2af0004
 .reloc 2, R_ARM_THM_ALU_PREL_11_0, dat1
/// adr.w r1, dat2
 .inst.w 0xf2af0104
 .reloc 6, R_ARM_THM_ALU_PREL_11_0, dat2
 nop
/// adr.w r2, dat3
 .inst.w 0xf2af0204
 .reloc 0xc, R_ARM_THM_ALU_PREL_11_0, dat3
/// adr.w r3, dat4
 .inst.w 0xf2af0304
 .reloc 0x10, R_ARM_THM_ALU_PREL_11_0, dat4
/// adr.w r0, target1
 .inst.w 0xf2af0004
 .reloc 0x14, R_ARM_THM_ALU_PREL_11_0, target1
 nop
/// adr.w r1, target2
 .inst.w 0xf2af0104
 .reloc 0x1a, R_ARM_THM_ALU_PREL_11_0, target2
 .section .text.pos, "ax", %progbits
 .balign 4
 .global pos
 .thumb_func
pos:
/// adr.w r2, target3
 .inst.w 0xf2af0204
 .reloc 0, R_ARM_THM_ALU_PREL_11_0, target3
 nop
/// adr.w r3, target4
 .inst.w 0xf2af0304
 .reloc 6, R_ARM_THM_ALU_PREL_11_0, target4
 nop
/// adr.w r0, dat5
 .inst.w 0xf2af0004
 .reloc 0xc, R_ARM_THM_ALU_PREL_11_0, dat5
/// adr.w r1, dat6
 .inst.w 0xf2af0104
 .reloc 0x10, R_ARM_THM_ALU_PREL_11_0, dat6

 nop
/// adr.w r2, dat7
 .inst.w 0xf2af0204
 .reloc 0x16, R_ARM_THM_ALU_PREL_11_0, dat7

/// adr.w r3, dat8
 .inst.w 0xf2af0304
 .reloc 0x1a, R_ARM_THM_ALU_PREL_11_0, dat8
/// positive addend in instruction, all others are -4 (PC bias)
/// adr.w r4, dat5 + 8
 .inst.w 0xf20f0404
 .reloc 0x1e, R_ARM_THM_ALU_PREL_11_0, dat5 + 8

 .section .text.high, "ax", %progbits
 .balign 4
 .thumb_func
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
// CHECK-NEXT:     8f00: bx      lr
// CHECK: 00008f02 <target2>:
// CHECK-NEXT:     8f02: bx      lr

// CHECK: 00009000 <_start>:
// CHECK-NEXT:     9000: nop
/// AlignDown(0x9002+4, 4) - 0xff2 = 0x8012
// CHECK-NEXT:     9002: adr.w   r0, #-4082
/// AlignDown(0x9006+4, 4) - 0xff5 = 0x8013
// CHECK-NEXT:     9006: adr.w   r1, #-4085
// CHECK-NEXT:     900a: nop
/// AlignDown(0x900c+4, 4) - 0xffc = 0x8014
// CHECK-NEXT:     900c: adr.w   r2, #-4092
/// AlignDown(0x9010+4, 4) - 0xfff = 0x8015
// CHECK-NEXT:     9010: adr.w   r3, #-4095
/// AlignDown(0x9014+4, 4) - 0x117 = 0x8f01
// CHECK-NEXT:     9014: adr.w   r0, #-279
// CHECK-NEXT:     9018: nop
/// AlignDown(0x901a+4, 4) - 0x119 = 0x8f03
// CHECK-NEXT:     901a: adr.w   r1, #-281

// CHECK: 00010000 <pos>:
/// AlignDown(0x10000+4, 4) + 0xfd = 0x10101
// CHECK-NEXT:    10000: adr.w   r2, #253
// CHECK-NEXT:    10004: nop
/// AlignDown(0x10006+4, 4) + 0xfb = 0x10103
// CHECK-NEXT:    10006: adr.w   r3, #251
// CHECK-NEXT:    1000a: nop
/// AlignDown(0x1000c+4, 4) + 0xfff = 0x1100f
// CHECK-NEXT:    1000c: adr.w   r0, #4095
/// AlignDown(0x10010+4, 4) + 0xffc = 0x11010
// CHECK-NEXT:    10010: adr.w   r1, #4092
// CHECK-NEXT:    10014: nop
/// AlignDown(0x10016+4, 4) + 0xff9 = 0x11011
// CHECK-NEXT:    10016: adr.w   r2, #4089
/// AlignDown(0x1001a+4, 4) + 0xff6 = 0x11012
// CHECK-NEXT:    1001a: adr.w   r3, #4086
/// AlignDown(0x1001e+4, 4) + 0xff7 = 0x11017 = dat5 + 8
// CHECK-NEXT:   1001e:  adr.w  r4, #4087

// CHECK: 00010100 <target3>:
// CHECK-NEXT:    10100: bx      lr

// CHECK: 00010102 <target4>:
// CHECK-NEXT:    10102: bx      lr

// SYMS:     Name: dat5
// SYMS-NEXT:     Value: 0x1100F
// SYMS:     Name: dat6
// SYMS-NEXT:     Value: 0x11010
// SYMS:     Name: dat7
// SYMS-NEXT:     Value: 0x11011
// SYMS:     Name: dat8
// SYMS-NEXT:     Value: 0x11012
