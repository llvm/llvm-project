// REQUIRES: aarch64
// RUN: llvm-mc -mattr=+bti -filetype=obj -triple=aarch64 %s -o %t.o
// RUN: echo "SECTIONS { .text 0x10000 : { *(.text.01); . += 0x8000000; *(.text.far); } }" > %t.script
// RUN: ld.lld -z force-bti --script %t.script -fix-cortex-a53-843419 -verbose %t.o -o %t2 \
// RUN:   2>&1 | FileCheck -check-prefix=CHECK-PRINT %s
// RUN: llvm-objdump --no-print-imm-hex --no-show-raw-insn --triple=aarch64-linux-gnu -d %t2 | FileCheck %s

/// Test case for specific crash wrt interaction between thunks where
/// relocations end up putting a BTI section in an unexpected position.
/// This case has been observed on a Chromium build and, although it is possible
/// to reproduce without the Cortex-A53 Erratum 843419 thunk, I kept it to
/// keep it as close as possible to the original situation.

.section .note.gnu.property,"a"
.p2align 3
.long 4
.long 0x10                   // descriptor length
.long 0x5                    // GNU property type
.asciz "GNU"
.long 0xc0000000             // GNU_PROPERTY_AARCH64_FEATURE_1_AND
.long 4
.long 1                      // GNU_PROPERTY_AARCH64_FEATURE_1_BTI
.long 0

        .section .text.01, "ax", %progbits
        .balign 4096
        .globl _start
        .type _start, %function
_start:
        bl far_away_no_bti

        .section .text.far, "ax", %progbits
        .globl far_away_no_bti
        .type far_away, function
far_away_no_bti:
        .space 4096 - 28, 0
        adrp x0, dat
        ldr x1, [x1, #0]
        ldr x0, [x0, :got_lo12:dat]
        .space 0x8000000, 0
        ret

        .section .data
        .globl dat
dat:    .quad 0

// CHECK-PRINT: detected cortex-a53-843419 erratum sequence starting at 8010FFC in unpatched output.

// Sanity check
// CHECK: 0000000000010000 <_start>:
// CHECK-NEXT: bl      0x10008 <__AArch64AbsLongThunk_far_away_no_bti>

// Check that the BTI thunks are kept small, they didn't moved and they do contain the landing pad
// CHECK: 0000000000010008 <__AArch64AbsLongThunk_far_away_no_bti>:
// CHECK: 0000000008010018 <__AArch64BTIThunk_far_away_no_bti>:
// CHECK-NEXT: bti     c
// CHECK: 0000000008010020 <__AArch64AbsLongThunk___CortexA53843419_8011004>:

// CHECK: 0000000008010030 <__AArch64BTIThunk___CortexA53843419_8011004_ret>:
// CHECK-NEXT: bti     c
// CHECK: 0000000008010038 <far_away_no_bti>:
// CHECK: b       0x8010020 <__AArch64AbsLongThunk___CortexA53843419_8011004>
// CHECK: 0000000008011028 <__CortexA53843419_8011004_ret>:

// Check that the errata thunk does NOT contain a landing pad
// CHECK: 000000001001102c <__CortexA53843419_8011004>:
// CHECK-NEXT: ldr     x0, [x0, #64]

