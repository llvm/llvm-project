// REQUIRES: arm
// RUN: llvm-mc -filetype=obj -triple=armv7a-linux-gnueabihf --arm-add-build-attributes %s -o %t.o
// RUN: ld.lld --fix-cortex-a8 -verbose %t.o -o /dev/null 2>&1 | FileCheck %s
/// Test that we warn, but don't attempt to patch when it is impossible to
/// redirect the branch as the Section is too large.

// CHECK: skipping cortex-a8 657417 erratum sequence, section .text is too large to patch
// CHECK: skipping cortex-a8 657417 erratum sequence, section .text.02 is too large to patch

 .syntax unified
 .thumb
/// Case 1: 1 MiB conditional branch range without relocation.
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
/// instruction, a patch will be attempted. Unfortunately the branch
/// cannot reach outside the section so we have to abort the patch.
 nop.w
 beq.w target
 .space 1024 * 1024

/// Case 2: 16 MiB
 .section .text.01, "ax", %progbits
 .balign 4096
  .space 4090
 .global target2
 .thumb_func
target2:
 .section .text.02, "ax", %progbits
/// 32-bit Branch and link spans 2 4KiB regions, preceded by a 32-bit
/// non branch instruction, a patch will be attempted. Unfortunately the
/// the BL cannot reach outside the section so we have to abort the patch.
 nop.w
 bl target2
 .space 16 * 1024 * 1024
