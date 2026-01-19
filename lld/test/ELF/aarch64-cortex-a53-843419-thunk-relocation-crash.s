// REQUIRES: aarch64
// RUN: rm -rf %t && split-file %s %t && cd %t
// RUN: llvm-mc -mattr=+bti -filetype=obj -triple=aarch64 asm -o a.o
// RUN: ld.lld --script lds -fix-cortex-a53-843419 -verbose a.o -o exe \
// RUN:   2>&1 | FileCheck -check-prefix=CHECK-PRINT %s
// RUN: llvm-objdump --no-print-imm-hex --no-show-raw-insn --triple=aarch64-linux-gnu -d exe | FileCheck %s

/// Test case for specific crash wrt interaction between thunks and errata
/// patches where the size of the added thunks meant that a range-extension
/// thunk to the patch was required. We need to check that a BTI Thunk is
/// generated for the patch, and that the patch's direct branch return is also
/// range extended, possibly needing another BTI Thunk.
///
/// The asm below is based on a crash that was happening in Chromium.
/// For more information see https://issues.chromium.org/issues/440019454

//--- asm
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

// CHECK: 0000000000010000 <_start>:
// CHECK-NEXT: 10000:       bl      0x10008 <__AArch64AbsLongThunk_far_away_no_bti>

// CHECK: <__AArch64AbsLongThunk_far_away_no_bti>:
// CHECK-NEXT: 10008:       ldr     x16, 0x10010
// CHECK-NEXT:              br      x16
// CHECK-NEXT: 10010: 18 00 01 08   .word   0x08010018

// Check that the BTI thunks do NOT have their size rounded up to 4 KiB.
// They precede the patch and they contain the landing pad.
// CHECK: <__AArch64BTIThunk_far_away_no_bti>:
// CHECK-NEXT: 8010018:       bti     c
// CHECK-NEXT:                b       0x8010038 <far_away_no_bti>

// CHECK: <__AArch64AbsLongThunk___CortexA53843419_8011004>:
// CHECK-NEXT: 8010020:       ldr     x16, 0x8010028
// CHECK-NEXT:                br      x16
// CHECK-NEXT: 8010028: 34 10 01 10   .word   0x10011034

// CHECK: <__AArch64BTIThunk_>:
// CHECK-NEXT: 8010030:       bti     c
// CHECK-NEXT:                b       0x8011028 <far_away_no_bti+0xff0>

// CHECK: 8010038 <far_away_no_bti>:
// CHECK-NEXT: ...
// CHECK-NEXT: 801101c:       adrp    x0, 0x10012000
// CHECK-NEXT:                ldr     x1, [x1]
// CHECK-NEXT:                b       0x8010020 <__AArch64AbsLongThunk___CortexA53843419_8011004>
// CHECK-NEXT: ...
// CHECK-NEXT: 10011028:       ret

// Check that the errata thunk does NOT contain a landing pad
// CHECK: <__CortexA53843419_8011004>:
// CHECK-NEXT: 1001102c:       ldr     x0, [x0, #64]
// CHECK-NEXT:                 b       0x10011040 <__AArch64AbsLongThunk_>

// Rest of generated code for readability
// CHECK: <__AArch64BTIThunk___CortexA53843419_8011004>:
// CHECK-NEXT: 10011034:       bti     c
// CHECK-NEXT:                 b       0x1001102c <__CortexA53843419_8011004>

// CHECK: <__AArch64AbsLongThunk_>
// CHECK-NEXT: 10011040:       ldr     x16, 0x10011048
// CHECK-NEXT:                 br      x16
// CHECK-NEXT: 10011048: 30 00 01 08   .word   0x08010030

//--- lds
SECTIONS {
  .text 0x10000 : {
    *(.text.01);
    . += 0x8000000;
    *(.text.far);
  }
}

