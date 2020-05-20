// REQUIRES: arm
// RUN: llvm-mc -filetype=obj -triple=armv7a-linux-gnueabihf --arm-add-build-attributes %s -o %t.o
// RUN: ld.lld --fix-cortex-a8 -verbose %t.o -o %t2
// RUN: llvm-objdump -d %t2 --start-address=0x21ffa --stop-address=0x22002 --no-show-raw-insn | FileCheck --check-prefix=CALLSITE1 %s
// RUN: llvm-objdump -d %t2 --start-address=0x22ffa --stop-address=0x23002 --no-show-raw-insn | FileCheck --check-prefix=CALLSITE2 %s
// RUN: llvm-objdump -d %t2 --start-address=0x23ffa --stop-address=0x24002 --no-show-raw-insn | FileCheck --check-prefix=CALLSITE3 %s
// RUN: llvm-objdump -d %t2 --start-address=0x24ffa --stop-address=0x25006 --no-show-raw-insn | FileCheck --check-prefix=CALLSITE4 %s
// RUN: llvm-objdump -d %t2 --start-address=0x25ffe --stop-address=0x26002 --no-show-raw-insn | FileCheck --check-prefix=CALLSITE5 %s
// RUN: llvm-objdump -d %t2 --start-address=0x27000 --stop-address=0x28004 --no-show-raw-insn | FileCheck --check-prefix=CALLSITE6 %s
// RUN: llvm-objdump -d %t2 --start-address=0x28002 --stop-address=0x29006 --no-show-raw-insn | FileCheck --check-prefix=CALLSITE7 %s

/// Test boundary conditions of the cortex-a8 erratum. The following cases
/// should not trigger the Erratum
 .syntax unified
 .thumb
 .text
 .global _start
 .balign 4096
 .thumb_func
_start:
 nop.w
 .space 4086
 .thumb_func
target:
/// 32-bit branch spans 2 4KiB regions, preceded by a 32-bit branch so no patch
/// expected.
 b.w target
 b.w target

// CALLSITE1:      00021ffa <target>:
// CALLSITE1-NEXT:    21ffa:            b.w     #-4
// CALLSITE1-NEXT:    21ffe:            b.w     #-8

 .space 4088
 .type target2, %function
target2:
/// 32-bit Branch and link spans 2 4KiB regions, preceded by a 16-bit
/// instruction so no patch expected.
 nop
 nop
 bl target2

// CALLSITE2:      00022ffa <target2>:
// CALLSITE2-NEXT:    22ffa:            nop
// CALLSITE2-NEXT:    22ffc:            nop
// CALLSITE2-NEXT:    22ffe:            bl      #-8

 .space 4088
 .type target3, %function
target3:
/// 32-bit conditional branch spans 2 4KiB regions, preceded by a 32-bit
/// non branch instruction, branch is backwards but outside 4KiB region. So
/// expect no patch.
 nop.w
 beq.w target2

// CALLSITE3:      00023ffa <target3>:
// CALLSITE3-NEXT:    23ffa:            nop.w
// CALLSITE3-NEXT:    23ffe:            beq.w   #-4104

 .space 4088
 .type source4, %function
source4:
/// 32-bit conditional branch spans 2 4KiB regions, preceded by a 32-bit
/// non branch instruction, branch is forwards to 2nd region so expect no patch.
 nop.w
 beq.w target4
 .thumb_func
target4:
 nop.w

// CALLSITE4:      00024ffa <source4>:
// CALLSITE4-NEXT:    24ffa:            nop.w
// CALLSITE4-NEXT:    24ffe:            beq.w   #0
// CALLSITE4:      00025002 <target4>:
// CALLSITE4-NEXT:    25002:            nop.w

 .space 4084
 .type target5, %function

target5:
/// 32-bit conditional branch spans 2 4KiB regions, preceded by the encoding of
/// a 32-bit thumb instruction, but in ARM state (illegal instruction), we
/// should not decode and match it as Thumb, expect no patch.
 .arm
 .inst 0x800f3af /// nop.w encoding in Thumb
 .thumb
 .thumb_func
source5:
 beq.w target5

// CALLSITE5:      00025ffe <source5>:
// CALLSITE5-NEXT:    25ffe:            beq.w   #-8

/// Edge case where two word sequence starts at offset 0xffc, check that
/// we don't match. In this case the branch will be completely in the 2nd
/// region and the branch will target the second region. This will pass a
/// branch destination in the same region test, but not the branch must have
/// and address of the form xxxxxffe.
 .space 4090
 .type target6, %function
 nop.w
/// Make sure target of branch is in the same 4KiB region as the branch.
target6:
 bl target6

// CALLSITE6:      00027000 <target6>:
// CALLSITE6-NEXT:    27000:            bl      #-4

/// Edge case where two word sequence starts at offset 0xffe, check that
/// we don't match. In this case the branch will be completely in the 2nd
/// region and the branch will target the second region. This will pass a
/// branch destination in the same region test, but not the branch must have
/// and address of the form xxxxxffe.
 .space 4090
 .type target7, %function
 nop.w
/// Make sure target of branch is in the same 4KiB region as the branch.
target7:
 bl target7

// CALLSITE7:      00028002 <target7>:
// CALLSITE7:         28002:            bl      #-4
