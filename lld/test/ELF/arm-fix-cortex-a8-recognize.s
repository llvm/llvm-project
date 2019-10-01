// REQUIRES: arm
// RUN: llvm-mc -filetype=obj -triple=armv7a-linux-gnueabihf --arm-add-build-attributes %s -o %t.o
// RUN: ld.lld --fix-cortex-a8 -verbose %t.o -o %t2 2>&1 | FileCheck %s
// RUN: llvm-objdump -d %t2 --start-address=0x1a004 --stop-address=0x1a024 --no-show-raw-insn | FileCheck --check-prefix=CHECK-PATCHES %s
// RUN: llvm-objdump -d %t2 --start-address=0x12ffa --stop-address=0x13002 --no-show-raw-insn | FileCheck --check-prefix=CALLSITE1 %s
// RUN: llvm-objdump -d %t2 --start-address=0x13ffa --stop-address=0x14002 --no-show-raw-insn | FileCheck --check-prefix=CALLSITE2 %s
// RUN: llvm-objdump -d %t2 --start-address=0x14ffa --stop-address=0x15002 --no-show-raw-insn | FileCheck --check-prefix=CALLSITE3 %s
// RUN: llvm-objdump -d %t2 --start-address=0x15ff4 --stop-address=0x16002 --no-show-raw-insn | FileCheck --check-prefix=CALLSITE4 %s
// RUN: llvm-objdump -d %t2 --start-address=0x16ffa --stop-address=0x17002 --no-show-raw-insn | FileCheck --check-prefix=CALLSITE5 %s
// RUN: llvm-objdump -d %t2 --start-address=0x17ffa --stop-address=0x18002 --no-show-raw-insn | FileCheck --check-prefix=CALLSITE6 %s
// RUN: llvm-objdump -d %t2 --start-address=0x18ffa --stop-address=0x19002 --no-show-raw-insn | FileCheck --check-prefix=CALLSITE7 %s
// RUN: llvm-objdump -d %t2 --start-address=0x19ff4 --stop-address=0x1a002 --no-show-raw-insn | FileCheck --check-prefix=CALLSITE8 %s

// CHECK:      ld.lld: detected cortex-a8-657419 erratum sequence starting at 12FFE in unpatched output.
// CHECK-NEXT: ld.lld: detected cortex-a8-657419 erratum sequence starting at 13FFE in unpatched output.
// CHECK-NEXT: ld.lld: detected cortex-a8-657419 erratum sequence starting at 14FFE in unpatched output.
// CHECK-NEXT: ld.lld: detected cortex-a8-657419 erratum sequence starting at 15FFE in unpatched output.
// CHECK-NEXT: ld.lld: detected cortex-a8-657419 erratum sequence starting at 16FFE in unpatched output.
// CHECK-NEXT: ld.lld: detected cortex-a8-657419 erratum sequence starting at 17FFE in unpatched output.
// CHECK-NEXT: ld.lld: detected cortex-a8-657419 erratum sequence starting at 18FFE in unpatched output.

/// Basic tests for the -fix-cortex-a8 erratum fix. The full details of the
/// erratum and the patch are in ARMA8ErrataFix.cpp . The test creates an
/// instance of the erratum every 4KiB (32-bit non-branch, followed by 32-bit
/// branch instruction, where the branch instruction spans two 4 KiB regions,
/// and the branch destination is in the first 4KiB region.
///
/// Test each 32-bit branch b.w, bcc.w, bl, blx. For b.w, bcc.w, and bl we
/// check the relocated and non-relocated forms. The blx instruction
/// always has a relocation in assembler.
 .syntax unified
 .thumb
 .text
 .global _start
 .type _start, %function
 .balign 4096
 .thumb_func
_start:
 nop.w
 .space 4086
 .thumb_func
 .global target
 .type target, %function
target:
/// 32-bit Branch spans 2 4KiB regions, preceded by a 32-bit non branch
/// instruction, expect a patch.
 nop.w
 b.w target

// CALLSITE1:      00012ffa target:
// CALLSITE1-NEXT:    12ffa:            nop.w
// CALLSITE1-NEXT:    12ffe:            b.w     #28674

 .space 4088
 .type target2, %function
 .local target2
target2:
/// 32-bit Branch and link spans 2 4KiB regions, preceded by a 32-bit
/// non branch instruction, expect a patch.
 nop.w
 bl target2

// CALLSITE2:      00013ffa target2:
// CALLSITE2-NEXT:    13ffa:            nop.w
// CALLSITE2-NEXT:    13ffe:            bl      #24582

 .space 4088
 .type target3, %function
 .local target3
target3:
/// 32-bit conditional branch spans 2 4KiB regions, preceded by a 32-bit
/// non branch instruction, expect a patch.
 nop.w
 beq.w target3

// CALLSITE3:      00014ffa target3:
// CALLSITE3-NEXT:    14ffa:            nop.w
// CALLSITE3-NEXT:    14ffe:            beq.w   #20490

 .space 4082
 .type target4, %function
 .local target4
 .arm
target4:
 bx lr
 .space 2
 .thumb
/// 32-bit Branch link and exchange spans 2 4KiB regions, preceded by a
/// 32-bit non branch instruction, blx always goes via relocation. Expect
/// a patch.
 nop.w
 blx target4

/// Target = 0x19010 __CortexA8657417_15FFE
// CALLSITE4:      00015ff4 target4:
// CALLSITE4-NEXT:    15ff4:            bx      lr
// CALLSITE4:         15ff8:    00 00           .short  0x0000
// CALLSITE4:         15ffa:            nop.w
// CALLSITE4-NEXT:    15ffe:            blx     #16400

/// Separate sections for source and destination of branches to force
/// a relocation.
 .section .text.0, "ax", %progbits
 .balign 2
 .global target5
 .type target5, %function
target5:
 nop.w
 .section .text.1, "ax", %progbits
 .space 4084
/// 32-bit branch spans 2 4KiB regions, preceded by a 32-bit non branch
/// instruction, expect a patch. Branch to global symbol so goes via a
/// relocation.
 nop.w
 b.w target5

/// Target = 0x19014 __CortexA8657417_16FFE
// CALLSITE5:         16ffa:            nop.w
// CALLSITE5-NEXT:    16ffe:            b.w     #12306

 .section .text.2, "ax", %progbits
 .balign 2
 .global target6
 .type target6, %function
target6:
 nop.w
 .section .text.3, "ax", %progbits
 .space 4084
/// 32-bit branch and link spans 2 4KiB regions, preceded by a 32-bit
/// non branch instruction, expect a patch. Branch to global symbol so
/// goes via a relocation.
 nop.w
 bl target6

/// Target = 0x19018 __CortexA8657417_17FFE
// CALLSITE6:         17ffa:            nop.w
// CALLSITE6-NEXT:    17ffe:            bl      #8214

 .section .text.4, "ax", %progbits
 .global target7
 .type target7, %function
target7:
 nop.w
 .section .text.5, "ax", %progbits
 .space 4084
/// 32-bit conditional branch spans 2 4KiB regions, preceded by a 32-bit
/// non branch instruction, expect a patch. Branch to global symbol so
/// goes via a relocation.
 nop.w
 bne.w target7

// CALLSITE7:         18ffa:            nop.w
// CALLSITE7-NEXT:    18ffe:            bne.w   #4122

 .section .text.6, "ax", %progbits
 .space 4082
 .arm
 .global target8
 .type target8, %function
target8:
 bx lr

 .section .text.7, "ax", %progbits
 .space 2
 .thumb
/// 32-bit Branch link spans 2 4KiB regions, preceded by a 32-bit non branch
/// instruction, expect a patch. The target of the BL is in ARM state so we
/// expect it to be turned into a BLX. The patch must be in ARM state to
/// avoid a state change thunk.
 nop.w
 bl target8

// CALLSITE8:      00019ff4 target8:
// CALLSITE8-NEXT:    19ff4:            bx      lr
// CALLSITE8:         19ff8:    00 00           .short  0x0000
// CALLSITE8:         19ffa:            nop.w
// CALLSITE8-NEXT:    19ffe:            blx     #32

// CHECK-PATCHES: 0001a004 __CortexA8657417_12FFE:
// CHECK-PATCHES-NEXT:    1a004:        b.w     #-28686

// CHECK-PATCHES:      0001a008 __CortexA8657417_13FFE:
// CHECK-PATCHES-NEXT:    1a008:        b.w     #-24594

// CHECK-PATCHES:      0001a00c __CortexA8657417_14FFE:
// CHECK-PATCHES-NEXT:    1a00c:        b.w     #-20502

// CHECK-PATCHES:      0001a010 __CortexA8657417_15FFE:
// CHECK-PATCHES-NEXT:    1a010:        b       #-16420

// CHECK-PATCHES:      0001a014 __CortexA8657417_16FFE:
// CHECK-PATCHES-NEXT:    1a014:        b.w     #-16406

// CHECK-PATCHES:      0001a018 __CortexA8657417_17FFE:
// CHECK-PATCHES-NEXT:    1a018:        b.w     #-12314

// CHECK-PATCHES:      0001a01c __CortexA8657417_18FFE:
// CHECK-PATCHES-NEXT:    1a01c:        b.w     #-8222

// CHECK-PATCHES:      0001a020 __CortexA8657417_19FFE:
// CHECK-PATCHES-NEXT:    1a020:        b       #-52
