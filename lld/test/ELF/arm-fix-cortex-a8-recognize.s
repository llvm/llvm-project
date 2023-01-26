// REQUIRES: arm
// RUN: llvm-mc -filetype=obj -triple=armv7a-linux-gnueabihf --arm-add-build-attributes %s -o %t.o
// RUN: ld.lld --fix-cortex-a8 -verbose %t.o -o %t2 2>&1 | FileCheck %s
// RUN: llvm-objdump --no-print-imm-hex -d %t2 --start-address=0x29004 --stop-address=0x29024 --no-show-raw-insn | FileCheck --check-prefix=CHECK-PATCHES %s
// RUN: llvm-objdump --no-print-imm-hex -d %t2 --start-address=0x21ffa --stop-address=0x22002 --no-show-raw-insn | FileCheck --check-prefix=CALLSITE1 %s
// RUN: llvm-objdump --no-print-imm-hex -d %t2 --start-address=0x22ffa --stop-address=0x23002 --no-show-raw-insn | FileCheck --check-prefix=CALLSITE2 %s
// RUN: llvm-objdump --no-print-imm-hex -d %t2 --start-address=0x23ffa --stop-address=0x24002 --no-show-raw-insn | FileCheck --check-prefix=CALLSITE3 %s
// RUN: llvm-objdump --no-print-imm-hex -d %t2 --start-address=0x24ff4 --stop-address=0x25002 --no-show-raw-insn | FileCheck --check-prefix=CALLSITE4 %s
// RUN: llvm-objdump --no-print-imm-hex -d %t2 --start-address=0x25ffa --stop-address=0x26002 --no-show-raw-insn | FileCheck --check-prefix=CALLSITE5 %s
// RUN: llvm-objdump --no-print-imm-hex -d %t2 --start-address=0x26ffa --stop-address=0x27002 --no-show-raw-insn | FileCheck --check-prefix=CALLSITE6 %s
// RUN: llvm-objdump --no-print-imm-hex -d %t2 --start-address=0x27ffa --stop-address=0x28002 --no-show-raw-insn | FileCheck --check-prefix=CALLSITE7 %s
// RUN: llvm-objdump --no-print-imm-hex -d %t2 --start-address=0x28ff4 --stop-address=0x29002 --no-show-raw-insn | FileCheck --check-prefix=CALLSITE8 %s
// RUN: ld.lld --fix-cortex-a8 -verbose -r %t.o -o %t3 2>&1 | FileCheck --check-prefix=CHECK-RELOCATABLE-LLD %s
// RUN: llvm-objdump --no-print-imm-hex --no-show-raw-insn -d %t3 --start-address=0xffa --stop-address=0x1002 | FileCheck --check-prefix=CHECK-RELOCATABLE %s

// CHECK:      ld.lld: detected cortex-a8-657419 erratum sequence starting at 22FFE in unpatched output.
// CHECK-NEXT: ld.lld: detected cortex-a8-657419 erratum sequence starting at 23FFE in unpatched output.
// CHECK-NEXT: ld.lld: detected cortex-a8-657419 erratum sequence starting at 24FFE in unpatched output.
// CHECK-NEXT: ld.lld: detected cortex-a8-657419 erratum sequence starting at 25FFE in unpatched output.
// CHECK-NEXT: ld.lld: detected cortex-a8-657419 erratum sequence starting at 26FFE in unpatched output.
// CHECK-NEXT: ld.lld: detected cortex-a8-657419 erratum sequence starting at 27FFE in unpatched output.
// CHECK-NEXT: ld.lld: detected cortex-a8-657419 erratum sequence starting at 28FFE in unpatched output.

/// We do not detect errors when doing a relocatable link as we don't know what
/// the final addresses are.
// CHECK-RELOCATABLE-LLD-NOT: ld.lld: detected cortex-a8-657419 erratum sequence

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

// CALLSITE1:      00021ffa <target>:
// CALLSITE1-NEXT:    21ffa:            nop.w
// CALLSITE1-NEXT:    21ffe:            b.w     0x29004 <__CortexA8657417_21FFE>
/// Expect no patch when doing a relocatable link ld -r.
// CHECK-RELOCATABLE: 00000ffa <target>:
// CHECK-RELOCATABLE-NEXT:      ffa:            nop.w
// CHECK-RELOCATABLE-NEXT:      ffe:            b.w     {{.+}} @ imm = #-4

 .space 4088
 .type target2, %function
 .local target2
target2:
/// 32-bit Branch and link spans 2 4KiB regions, preceded by a 32-bit
/// non branch instruction, expect a patch.
 nop.w
 bl target2

// CALLSITE2:      00022ffa <target2>:
// CALLSITE2-NEXT:    22ffa:            nop.w
// CALLSITE2-NEXT:    22ffe:            bl      0x29008 <__CortexA8657417_22FFE>

 .space 4088
 .type target3, %function
 .local target3
target3:
/// 32-bit conditional branch spans 2 4KiB regions, preceded by a 32-bit
/// non branch instruction, expect a patch.
 nop.w
 beq.w target3

// CALLSITE3:      00023ffa <target3>:
// CALLSITE3-NEXT:    23ffa:            nop.w
// CALLSITE3-NEXT:    23ffe:            beq.w   0x2900c <__CortexA8657417_23FFE>

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
// CALLSITE4:      00024ff4 <target4>:
// CALLSITE4-NEXT:    24ff4:            bx      lr
// CALLSITE4:         24ff8:    00 00           .short  0x0000
// CALLSITE4:         24ffa:            nop.w
// CALLSITE4-NEXT:    24ffe:            blx     0x29010 <__CortexA8657417_24FFE>

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
// CALLSITE5:         25ffa:            nop.w
// CALLSITE5-NEXT:    25ffe:            b.w     0x29014 <__CortexA8657417_25FFE>

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
// CALLSITE6:         26ffa:            nop.w
// CALLSITE6-NEXT:    26ffe:            bl      0x29018 <__CortexA8657417_26FFE>

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

// CALLSITE7:         27ffa:            nop.w
// CALLSITE7-NEXT:    27ffe:            bne.w   0x2901c <__CortexA8657417_27FFE>

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

// CALLSITE8:      00028ff4 <target8>:
// CALLSITE8-NEXT:    28ff4:            bx      lr
// CALLSITE8:         28ff8:    00 00           .short  0x0000
// CALLSITE8:         28ffa:            nop.w
// CALLSITE8-NEXT:    28ffe:            blx     0x29020 <__CortexA8657417_28FFE>

// CHECK-PATCHES: 00029004 <__CortexA8657417_21FFE>:
// CHECK-PATCHES-NEXT:    29004:        b.w     0x21ffa <target>

// CHECK-PATCHES:      00029008 <__CortexA8657417_22FFE>:
// CHECK-PATCHES-NEXT:    29008:        b.w     0x22ffa <target2>

// CHECK-PATCHES:      0002900c <__CortexA8657417_23FFE>:
// CHECK-PATCHES-NEXT:    2900c:        b.w     0x23ffa <target3>

// CHECK-PATCHES:      00029010 <__CortexA8657417_24FFE>:
// CHECK-PATCHES-NEXT:    29010:        b       0x24ff4 <target4>

// CHECK-PATCHES:      00029014 <__CortexA8657417_25FFE>:
// CHECK-PATCHES-NEXT:    29014:        b.w     0x25002 <target5>

// CHECK-PATCHES:      00029018 <__CortexA8657417_26FFE>:
// CHECK-PATCHES-NEXT:    29018:        b.w     0x26002 <target6>

// CHECK-PATCHES:      0002901c <__CortexA8657417_27FFE>:
// CHECK-PATCHES-NEXT:    2901c:        b.w     0x27002 <target7>

// CHECK-PATCHES:      00029020 <__CortexA8657417_28FFE>:
// CHECK-PATCHES-NEXT:    29020:        b       0x28ff4 <target8>
