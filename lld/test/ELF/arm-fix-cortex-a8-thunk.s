// REQUIRES: arm
// RUN: llvm-mc -filetype=obj -triple=armv7a-linux-gnueabihf --arm-add-build-attributes %s -o %t.o
// RUN: echo "SECTIONS { \
// RUN:          .text0 0x011006 : { *(.text.00) } \
// RUN:          .text1 0x110000 : { *(.text.01) *(.text.02) *(.text.03) \
// RUN:                             *(.text.04) } \
// RUN:          .text2 0x210000 : { *(.text.05) } } " > %t.script
// RUN: ld.lld --script %t.script --fix-cortex-a8 --shared -verbose %t.o -o %t2 2>&1
// RUN: llvm-objdump --no-print-imm-hex -d --no-show-raw-insn %t2 | FileCheck %s

/// Test cases for Cortex-a8 Erratum 657417 that involve interactions with
/// range extension thunks. Both erratum fixes and range extension thunks need
/// precise information and after creation alter address information.
 .thumb

 .section .text.00, "ax", %progbits
 .thumb_func
early:
 bx lr

 .section .text.01, "ax", %progbits
 .balign 4096
 .globl _start
 .type _start, %function
_start:
  beq.w far_away
/// Thunk to far_away and state change needed, size 12-bytes goes here.
// CHECK: 00110004 <__ThumbV7PILongThunk_far_away>:
// CHECK-NEXT: 110004: movw    r12, #65524
// CHECK-NEXT:         movt    r12, #15
// CHECK-NEXT:         add     r12, pc
// CHECK-NEXT:         bx      r12

 .section .text.02, "ax", %progbits
 .space 4096 - 22

 .section .text.03, "ax", %progbits
 .thumb_func
target:
/// After thunk is added this branch will line up across 2 4 KiB regions
/// and will trigger a patch.
 nop.w
 bl target

/// Expect erratum patch inserted here
// CHECK: 00110ffa <target>:
// CHECK-NEXT: 110ffa: nop.w
// CHECK-NEXT:         bl      0x111004 <__CortexA8657417_110FFE>
// CHECK: 00111004 <__CortexA8657417_110FFE>:
// CHECK-NEXT: 111004: b.w     0x110ffa <target>

/// Expect range extension thunk here.
// CHECK: 00111008 <__ThumbV7PILongThunk_early>:
// CHECK-NEXT: 111008: b.w     0x11006 <early>

 .section .text.04, "ax", %progbits
/// The erratum patch will push this branch out of range, so another
/// range extension thunk will be needed.
        beq.w early
// CHECK:   11100c:            beq.w   0x111008 <__ThumbV7PILongThunk_early>

 .section .text.05, "ax", %progbits
 .arm
 nop
 .type far_away, %function
far_away:
  bx lr
