// REQUIRES: arm
// RUN: llvm-mc -filetype=obj -triple=armv7a-unknown-linux-gnueabi %s -o %t
// RUN: ld.lld %t -o %t2
// RUN: llvm-objdump -d %t2 --triple=armv7a-unknown-linux-gnueabi --no-show-raw-insn | FileCheck %s
// RUN: llvm-mc -filetype=obj -triple=thumbv7a-unknown-linux-gnueabi %s -o %t3
// RUN: ld.lld %t3 -o %t4
// RUN: llvm-objdump -d %t4 --triple=thumbv7a-unknown-linux-gnueabi --no-show-raw-insn | FileCheck %s

/// Test the following relocation pairs:
///  * R_ARM_MOVW_ABS_NC and R_ARM_MOVT_ABS
///  * R_ARM_MOVW_PREL_NC and R_ARM_MOVT_PREL
///  * R_ARM_MOVW_BREL_NC and R_ARM_MOVT_BREL
///
///  * R_ARM_THM_MOVW_ABS_NC and R_ARM_THM_MOVT_ABS
///  * R_ARM_THM_MOVW_PREL_NC and R_ARM_THM_MOVT_PREL
///  * R_ARM_THM_MOVW_BREL_NC and R_ARM_THM_MOVT_BREL

 .syntax unified
 .globl _start
 .align 12
_start:
 .section .R_ARM_MOVW_ABS_NC, "ax",%progbits
 .align 8
 movw r0, :lower16:label
 movw r1, :lower16:label1
 movw r2, :lower16:label2 + 4
 movw r3, :lower16:label3
 movw r4, :lower16:label3 + 4
// CHECK-LABEL: Disassembly of section .R_ARM_MOVW_ABS_NC
// CHECK-EMPTY:
// CHECK: 21000: movw    r0, #0
// CHECK:        movw    r1, #4
// CHECK:        movw    r2, #12
// CHECK:        movw    r3, #65532
/// :lower16:label3 + 4 = :lower16:0x30000 = 0
// CHECK:        movw    r4, #0

 .section .R_ARM_MOVT_ABS, "ax",%progbits
 .align 8
 movt r0, :upper16:label
 movt r1, :upper16:label1
 movt r2, :upper16:label2 + 4
 movt r3, :upper16:label3
 movt r4, :upper16:label3 + 4
// CHECK-LABEL: Disassembly of section .R_ARM_MOVT_ABS
// CHECK-EMPTY:
// CHECK: 21100: movt    r0, #4
// CHECK:        movt    r1, #4
// CHECK:        movt    r2, #4
// CHECK:        movt    r3, #4
/// :upper16:label3 + 4 = :upper16:0x30000 = 3
// CHECK:        movt    r4, #5

.section .R_ARM_MOVW_PREL_NC, "ax",%progbits
.align 8
 movw r0, :lower16:label - .
 movw r1, :lower16:label1 - .
 movw r2, :lower16:label2 + 4 - .
 movw r3, :lower16:label3 - .
 movw r4, :lower16:label3 + 0x2214 - .
// CHECK-LABEL: Disassembly of section .R_ARM_MOVW_PREL_NC
// CHECK-EMPTY:
/// :lower16:label - . = 56832
// CHECK: 21200: movw    r0, #60928
/// :lower16:label1 - . = 56832
// CHECK:        movw    r1, #60928
/// :lower16:label2 - . + 4 = 60932
// CHECK:        movw    r2, #60932
/// :lower16:label3 - . = 60912
// CHECK:        movw    r3, #60912
/// :lower16:label3 - . + 0x2214 = :lower16:0x20000 = 4096
// CHECK:        movw    r4, #4096

.section .R_ARM_MOVT_PREL, "ax",%progbits
.align 8
 movt r0, :upper16:label - .
 movt r1, :upper16:label1 - .
 movt r2, :upper16:label2 + 0x4 - .
 movt r3, :upper16:label3 - .
 movt r4, :upper16:label3 + 0x2314 - .
// CHECK-LABEL: Disassembly of section .R_ARM_MOVT_PREL
// CHECK-EMPTY:
/// :upper16:label - . = :upper16:0xdd00  = 1
// CHECK: 21300: movt    r0, #1
/// :upper16:label1 - . = :upper16:0xdd00 = 1
// CHECK:        movt    r1, #1
/// :upper16:label2 - . + 4 = :upper16:0xdd04 = 1
// CHECK:        movt    r2, #1
/// :upper16:label3 - . = :upper16:0x1dcf0 = 2
// CHECK:        movt    r3, #2
/// :upper16:label3 - . + 0x2314 = :upper16:0x20000 = 3
// CHECK:        movt    r4, #3

.section .R_ARM_MOVW_BREL_NC, "ax",%progbits
.align 8
 movw r0, :lower16:label(sbrel)
 movw r1, :lower16:label1(sbrel)
 movw r2, :lower16:label2(sbrel)
 movw r3, :lower16:label3(sbrel)
 movw r4, :lower16:label3.4(sbrel)
// CHECK-LABEL: Disassembly of section .R_ARM_MOVW_BREL_NC
// CHECK-EMPTY:
// SB = .destination
/// :lower16:label - SB = 0
// CHECK: 21400: movw    r0, #0
/// :lower16:label1 - SB = 4
// CHECK:        movw    r1, #4
/// :lower16:label2 - SB = 8
// CHECK:        movw    r2, #8
/// :lower16:label3 - SB = 0xfffc
// CHECK:        movw    r3, #65532
/// :lower16:label3.4 - SB = :lower16:0x10000 = 0
// CHECK:        movw    r4, #0

.section .R_ARM_MOVT_BREL, "ax",%progbits
.align 8
 movt r0, :upper16:label(sbrel)
 movt r1, :upper16:label1(sbrel)
 movt r2, :upper16:label2(sbrel)
 movt r3, :upper16:label3(sbrel)
 movt r4, :upper16:label3.4(sbrel)
// CHECK-LABEL: Disassembly of section .R_ARM_MOVT_BREL
// CHECK-EMPTY:
// SB = .destination
/// :upper16:label - SB = 0
// CHECK: 21500: movt    r0, #0
/// :upper16:label1 - SB = 0
// CHECK:        movt    r1, #0
/// :upper16:label2 - SB = 0
// CHECK:        movt    r2, #0
/// :upper16:label3 - SB = 0
// CHECK:        movt    r3, #0
/// :upper16:label3.4 - SB = :upper16:0x10000 = 1
// CHECK:        movt    r4, #1

.section .R_ARM_THM_MOVW_ABS_NC, "ax",%progbits
.align 8
 movw r0, :lower16:label
 movw r1, :lower16:label1
 movw r2, :lower16:label2 + 4
 movw r3, :lower16:label3
 movw r4, :lower16:label3 + 4
// CHECK-LABEL: Disassembly of section .R_ARM_THM_MOVW_ABS_NC
// CHECK-EMPTY:
// CHECK: 21600: movw    r0, #0
// CHECK:        movw    r1, #4
// CHECK:        movw    r2, #12
// CHECK:        movw    r3, #65532
// CHECK:        movw    r4, #0

.section .R_ARM_THM_MOVT_ABS, "ax",%progbits
.align 8
 movt r0, :upper16:label
 movt r1, :upper16:label1
 movt r2, :upper16:label2 + 4
 movt r3, :upper16:label3
 movt r4, :upper16:label3 + 4
// CHECK-LABEL: Disassembly of section .R_ARM_THM_MOVT_ABS
// CHECK-EMPTY:
// CHECK: 21700: movt    r0, #4
// CHECK:        movt    r1, #4
// CHECK:        movt    r2, #4
// CHECK:        movt    r3, #4
// CHECK:        movt    r4, #5

.section .R_ARM_THM_MOVW_PREL_NC, "ax",%progbits
.align 8
 movw r0, :lower16:label - .
 movw r1, :lower16:label1 - .
 movw r2, :lower16:label2 + 4 - .
 movw r3, :lower16:label3 - .
 movw r4, :lower16:label3 + 0x2814 - .
// CHECK-LABEL: Disassembly of section .R_ARM_THM_MOVW_PREL_NC
// CHECK-EMPTY:
/// :lower16:label - . = 59392
// CHECK: 21800: movw    r0, #59392
/// :lower16:label1 - . = 59392
// CHECK:        movw    r1, #59392
/// :lower16:label2 - . + 4 = 59396
// CHECK:        movw    r2, #59396
/// :lower16:label3 - . = 59376
// CHECK:        movw    r3, #59376
/// :lower16:label3 - . + 0x2814 = 0x20000
// CHECK:        movw    r4, #4096

.section .R_ARM_THM_MOVT_PREL, "ax",%progbits
.align 8
 movt r0, :upper16:label - .
 movt r1, :upper16:label1 - .
 movt r2, :upper16:label2 + 0x4 - .
 movt r3, :upper16:label3 - .
 movt r4, :upper16:label3 + 0x2914 - .
// CHECK-LABEL: Disassembly of section .R_ARM_THM_MOVT_PREL
// CHECK-EMPTY:
/// :upper16:label - . = :upper16:0xd700  = 1
// CHECK: 21900: movt    r0, #1
/// :upper16:label1 - . = :upper16:0xd700 = 1
// CHECK:        movt    r1, #1
/// :upper16:label2 - . + 4 = :upper16:0xd704 = 1
// CHECK:        movt    r2, #1
/// :upper16:label3 - . = :upper16:0x1d6f0 = 2
// CHECK:        movt    r3, #2
/// :upper16:label3 - . + 0x2914 = :upper16:0x20000 = 3
// CHECK:        movt    r4, #3

.section .R_ARM_THM_MOVW_BREL_NC, "ax",%progbits
.align 8
 movw r0, :lower16:label(sbrel)
 movw r1, :lower16:label1(sbrel)
 movw r2, :lower16:label2(sbrel)
 movw r3, :lower16:label3(sbrel)
 movw r4, :lower16:label3.4(sbrel)
// CHECK-LABEL: Disassembly of section .R_ARM_THM_MOVW_BREL_NC
// CHECK-EMPTY:
// SB = .destination
/// :lower16:label - SB = 0
// CHECK: 21a00: movw    r0, #0
/// :lower16:label1 - SB = 4
// CHECK:        movw    r1, #4
/// :lower16:label2 - SB = 8
// CHECK:        movw    r2, #8
/// :lower16:label3 - SB = 0xfffc
// CHECK:        movw    r3, #65532
/// :lower16:label3.4 - SB = :lower16:0x10000 = 0
// CHECK:        movw    r4, #0

.section .R_ARM_THM_MOVT_BREL, "ax",%progbits
.align 8
 movt r0, :upper16:label(sbrel)
 movt r1, :upper16:label1(sbrel)
 movt r2, :upper16:label2(sbrel)
 movt r3, :upper16:label3(sbrel)
 movt r4, :upper16:label3.4(sbrel)
// CHECK-LABEL: Disassembly of section .R_ARM_THM_MOVT_BREL
// CHECK-EMPTY:
/// SB = .destination
/// :upper16:label - SB = 0
// CHECK: 21b00: movt    r0, #0
/// :upper16:label1 - SB = 0
// CHECK:        movt    r1, #0
/// :upper16:label2 - SB = 0
// CHECK:        movt    r2, #0
/// :upper16:label3 - SB = 0
// CHECK:        movt    r3, #0
/// :upper16:label3.4 - SB = :upper16:0x10000 = 1
// CHECK:        movt    r4, #1

 .section .destination, "aw",%progbits
 .balign 65536
/// 0x20000
label:
 .word 0
/// 0x20004
label1:
 .word 1
/// 0x20008
label2:
 .word 2
/// Test label3 is immediately below 2^16 alignment boundary
 .space 65536 - 16
/// 0x2fffc
label3:
 .word 3
/// label3 + 4 is on a 2^16 alignment boundary
label3.4:
 .word 4
