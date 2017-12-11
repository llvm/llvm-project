// REQUIRES: aarch64
// RUN: llvm-mc -filetype=obj -triple=aarch64-none-linux %s -o %t.o
// RUN: echo "SECTIONS { \
// RUN:          .text : { *(.text) *(.text.*) *(.newisd) } \
// RUN:          .text2 : { *.(newos) } \
// RUN:          .data : { *(.data) } }" > %t.script
// RUN: ld.lld --script %t.script -fix-cortex-a53-843419 -verbose %t.o -o %t2 | FileCheck %s

// Test cases for Cortex-A53 Erratum 843419 that involve interactions
// between the generated patches and the address of sections.

// See ARM-EPM-048406 Cortex_A53_MPCore_Software_Developers_Errata_Notice.pdf
// for full erratum details.
// In Summary
// 1.)
// ADRP (0xff8 or 0xffc).
// 2.)
// - load or store single register or either integer or vector registers.
// - STP or STNP of either vector or vector registers.
// - Advanced SIMD ST1 store instruction.
// - Must not write Rn.
// 3.) optional instruction, can't be a branch, must not write Rn, may read Rn.
// 4.) A load or store instruction from the Load/Store register unsigned
// immediate class using Rn as the base register.

// An aarch64 section can contain ranges of literal data embedded within the
// code, these ranges are encoded with mapping symbols. This tests that we
// can match the erratum sequence in code, but not data.
// - We can handle more than one patch per code range (denoted by mapping
//   symbols).
// - We can handle a patch in more than range of code, with literal data
//   inbetween.
// - We can handle redundant mapping symbols (two or more consecutive mapping
//   symbols with the same type).
// - We can ignore erratum sequences in multiple literal data ranges.

// CHECK: detected cortex-a53-843419 erratum sequence starting at FF8 in unpatched output.

        .section .text.01, "ax", %progbits
        .balign 4096
        .space 4096 - 8
        .globl t3_ff8_ldr
        .type t3_ff8_ldr, %function
t3_ff8_ldr:
        adrp x0, dat
        ldr x1, [x1, #0]
        ldr x0, [x0, :got_lo12:dat]
        ret

        // create a redundant mapping symbol as we are already in a $x range
        // some object producers unconditionally generate a mapping symbol on
        // every symbol so we need to handle the case of $x $x.
        .local $x.999
$x.999:
// CHECK-NEXT: detected cortex-a53-843419 erratum sequence starting at 1FFC in unpatched output.
        .globl t3_ffc_ldrsimd
        .type t3_ffc_ldrsimd, %function
        .space 4096 - 12
t3_ffc_ldrsimd:
        adrp x0, dat
        ldr s1, [x1, #0]
        ldr x2, [x0, :got_lo12:dat]
        ret

// Inline data containing bit pattern of erratum sequence, expect no patch.
        .globl t3_ffc_ldralldata
        .type t3_ff8_ldralldata, %function
        .space 4096 - 20
t3_ff8_ldralldata:
        // 0x90000000 = adrp x0, #0
        .byte 0x00
        .byte 0x00
        .byte 0x00
        .byte 0x90
        // 0xf9400021 = ldr x1, [x1]
        .byte 0x21
        .byte 0x00
        .byte 0x40
        .byte 0xf9
        // 0xf9400000 = ldr x0, [x0]
        .byte 0x00
        .byte 0x00
        .byte 0x40
        .byte 0xf9
        // Check that we can recognise the erratum sequence post literal data.

// CHECK-NEXT: detected cortex-a53-843419 erratum sequence starting at 3FF8 in unpatched output.

        .space 4096 - 12
        .globl t3_ffc_ldr
        .type t3_ffc_ldr, %function
 t3_ffc_ldr:
        adrp x0, dat
        ldr x1, [x1, #0]
        ldr x0, [x0, :got_lo12:dat]
        ret

        .section .text.02, "ax", %progbits
        .space 4096 - 12

        // Start a new InputSectionDescription (see Linker Script) so the
        // start address will be affected by any patches added to previous
        // InputSectionDescription.

// CHECK: detected cortex-a53-843419 erratum sequence starting at 4FFC in unpatched output.

        .section .newisd, "ax", %progbits
        .globl t3_ffc_str
        .type t3_ffc_str, %function
t3_ffc_str:
        adrp x0, dat
        str x1, [x1, #0]
        ldr x0, [x0, :got_lo12:dat]
        ret
        .space 4096 - 20

// CHECK:  detected cortex-a53-843419 erratum sequence starting at 5FF8 in unpatched output.

        // Start a new OutputSection (see Linker Script) so the
        // start address will be affected by any patches added to previous
        // InputSectionDescription.
        .section .newos, "ax", %progbits
        .globl t3_ff8_str
        .type t3_ff8_str, %function
t3_ff8_str:
        adrp x0, dat
        str x1, [x1, #0]
        ldr x0, [x0, :got_lo12:dat]
        ret
        .globl _start
        .type _start, %function
_start:
        ret

        .data
        .globl dat
dat:    .word 0
