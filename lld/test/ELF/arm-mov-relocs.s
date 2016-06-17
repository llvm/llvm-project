// RUN: llvm-mc -filetype=obj -triple=armv7a-unknown-linux-gnueabi %s -o %t
// RUN: ld.lld %t -o %t2
// RUN: llvm-objdump -d %t2 -triple=armv7a-unknown-linux-gnueabi | FileCheck %s
// RUN: llvm-mc -filetype=obj -triple=thumbv7a-unknown-linux-gnueabi %s -o %t3
// RUN: ld.lld %t3 -o %t4
// RUN: llvm-objdump -d %t4 -triple=thumbv7a-unknown-linux-gnueabi | FileCheck %s
// REQUIRES: arm

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
// CHECK: movw	r0, #0
// CHECK: movw	r1, #4
// CHECK: movw	r2, #12
// CHECK: movw	r3, #65532
// CHECK: movw	r4, #0
 .section .R_ARM_MOVT_ABS, "ax",%progbits
 movt r0, :upper16:label
 movt r1, :upper16:label1
// FIXME: We shouldn't need to multiply by 65536.
// arguably llvm-mc incorrectly assembles addends for
// SHT_REL relocated movt instructions. When there is a relocation
// the interpretation of the addend for SHT_REL is not shifted
 movt r2, :upper16:label2 + (4 * 65536)
 movt r3, :upper16:label3
// FIXME: We shouldn't need to multiply by 65536 see comment above.
 movt r4, :upper16:label3 + (4 * 65536)
// CHECK: Disassembly of section .R_ARM_MOVT_ABS
// CHECK: movt	r0, #2
// CHECK: movt	r1, #2
// CHECK: movt	r2, #2
// CHECK: movt	r3, #2
// CHECK: movt	r4, #3

 .section .destination, "aw",%progbits
 .balign 65536
label:
 .word 0
label1:
 .word 1
label2:
 .word 2
// Test label3 is immediately below 2^16 alignment boundary
 .space 65536 - 16
label3:
 .word 3
// label3 + 4 is on a 2^16 alignment boundary
 .word 4
