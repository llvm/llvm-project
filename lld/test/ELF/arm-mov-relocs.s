// REQUIRES: arm
// RUN: llvm-mc -filetype=obj -triple=armv7a-unknown-linux-gnueabi %s -o %t
// RUN: ld.lld %t -o %t2
// RUN: llvm-objdump -d %t2 -triple=armv7a-unknown-linux-gnueabi --no-show-raw-insn | FileCheck %s
// RUN: llvm-mc -filetype=obj -triple=thumbv7a-unknown-linux-gnueabi %s -o %t3
// RUN: ld.lld %t3 -o %t4
// RUN: llvm-objdump -d %t4 -triple=thumbv7a-unknown-linux-gnueabi --no-show-raw-insn | FileCheck %s

// Test the R_ARM_MOVW_ABS_NC and R_ARM_MOVT_ABS relocations as well as
// the R_ARM_THM_MOVW_ABS_NC and R_ARM_THM_MOVT_ABS relocations.
 .syntax unified
 .globl _start
_start:
 .section .R_ARM_MOVW_ABS_NC, "ax",%progbits
 movw r0, :lower16:label
 movw r1, :lower16:label1
 movw r2, :lower16:label2 + 4
 movw r3, :lower16:label3
 movw r4, :lower16:label3 + 4
// CHECK: Disassembly of section .R_ARM_MOVW_ABS_NC
// CHECK-EMPTY:
// CHECK: movw	r0, #0
// CHECK: movw	r1, #4
// CHECK: movw	r2, #12
// CHECK: movw	r3, #65532
// CHECK: movw	r4, #0
 .section .R_ARM_MOVT_ABS, "ax",%progbits
 movt r0, :upper16:label
 movt r1, :upper16:label1
 movt r2, :upper16:label2 + 4
 movt r3, :upper16:label3
 movt r4, :upper16:label3 + 4
// CHECK: Disassembly of section .R_ARM_MOVT_ABS
// CHECK-EMPTY:
// CHECK: movt	r0, #2
// CHECK: movt	r1, #2
// CHECK: movt	r2, #2
// CHECK: movt	r3, #2
// CHECK: movt	r4, #3

.section .R_ARM_MOVW_PREL_NC, "ax",%progbits
 movw r0, :lower16:label - .
 movw r1, :lower16:label1 - .
 movw r2, :lower16:label2 + 4 - .
 movw r3, :lower16:label3 - .
 movw r4, :lower16:label3 + 0x103c - .
// 0x20000 - . = 61188
// CHECK: 110fc:       movw    r0, #61188
// 0x20004 - . = 61188
// CHECK: 11100:       movw    r1, #61188
// 0x20008 - . + 4 = 61192
// CHECK: 11104:       movw    r2, #61192
// 0x2fffc - . = 61172
// CHECK: 11108:       movw    r3, #61172
// 0x2fffc - . +0x103c = 65324
// CHECK: 1110c:       movw    r4, #65324

.section .R_ARM_MOVT_PREL, "ax",%progbits
 movt r0, :upper16:label - .
 movt r1, :upper16:label1 - .
 movt r2, :upper16:label2 + 0x4 - .
 movt r3, :upper16:label3 - .
 movt r4, :upper16:label3 + 0x1120 - .
// 0x20000 - . = :upper16:0xeef0  = 0
// CHECK: 11110:       movt    r0, #0
// 0x20004 - . = :upper16:0xeef0 = 0
// CHECK: 11114:       movt    r1, #0
// 0x20008 - . + 4 = :upper16:0xeef4 = 0
// CHECK: 11118:       movt    r2, #0
// 0x2fffc - . = :upper16:0x1eee0 = 1
// CHECK: 1111c:       movt    r3, #1
// 0x2fffc - . + 0x1120 = :upper16:0x20000 = 2
// CHECK: 11120:       movt    r4, #1
 .section .destination, "aw",%progbits
 .balign 65536
// 0x20000
label:
 .word 0
// 0x20004
label1:
 .word 1
// 0x20008
label2:
 .word 2
// Test label3 is immediately below 2^16 alignment boundary
 .space 65536 - 16
// 0x2fffc
label3:
 .word 3
// label3 + 4 is on a 2^16 alignment boundary
 .word 4
