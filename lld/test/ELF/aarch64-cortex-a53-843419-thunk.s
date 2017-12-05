// REQUIRES: aarch64
// RUN: llvm-mc -filetype=obj -triple=aarch64-none-linux %s -o %t.o
// RUN: echo "SECTIONS { \
// RUN:          .text1 0x10000 : { *(.text.01) *(.text.02) *(.text.03) } \
// RUN:          .text2 0x100000000 : { *(.text.04) } } " > %t.script
// RUN: ld.lld --script %t.script -fix-cortex-a53-843419 -verbose %t.o -o %t2 | FileCheck -check-prefix=CHECK-PRINT %s

// Test cases for Cortex-A53 Erratum 843419 that involve interactions with
// range extension thunks. Both erratum fixes and range extension thunks need
// precise address information and after creation alter address information.


        .section .text.01, "ax", %progbits
        .balign 4096
        .globl _start
        .type _start, %function
_start:
        bl far_away
        // Thunk to far_away, size 16-bytes goes here.

        .section .text.02, "ax", %progbits
        .space 4096 - 28

        // Erratum sequence will only line up at address 0 modulo 0xffc when
        // Thunk is inserted.
        .section .text.03, "ax", %progbits
        .globl t3_ff8_ldr
        .type t3_ff8_ldr, %function
t3_ff8_ldr:
        adrp x0, dat
        ldr x1, [x1, #0]
        ldr x0, [x0, :got_lo12:dat]
        ret

// CHECK-PRINT: detected cortex-a53-843419 erratum sequence starting at 10FFC in unpatched output.
        .section .text.04, "ax", %progbits
        .globl far_away
        .type far_away, function
far_away:
        ret

        .section .data
        .globl dat
dat:    .quad 0
