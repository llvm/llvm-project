// REQUIRES: arm
// RUN: rm -rf %t && split-file %s %t && cd %t
// RUN: llvm-mc -arm-add-build-attributes -filetype=obj -triple=armv4t-none-eabi a.s -o a.o
// RUN: ld.lld a.o -T overlay.ld -o overlay
// RUN: llvm-objdump -d --no-show-raw-insn overlay | FileCheck %s

/// A thunk in a different overlay should not be shared as we cannot guarantee it
/// is in memory. It is OK for an overlay to share a thunk in a non-overlay as
/// that will be in memory.

// CHECK-LABEL: <_start>:
// CHECK-NEXT: 1000: bl 0x1004
// CHECK-LABEL: <__Thumbv4ABSLongThunk_far>:

// CHECK-LABEL: <over1>:
// CHECK-NEXT: 2000: bl 0x1004
// CHECK-NEXT:       bl 0x200c
// CHECK-NEXT:       bl 0x200c
// CHECK-LABEL: <__Thumbv4ABSLongThunk_far2>:

// CHECK-LABEL: <over2>:
// CHECK-NEXT: 2000: bl 0x1004
// CHECK-NEXT:       bl 0x2010
// CHECK-NEXT:       bl 0x2010
// CHECK-LABEL: <__Thumbv4ABSLongThunk_far2>:

// CHECK-LABEL: <nonover>:
// CHECK-NEXT: 3000: bl 0x3004
// CHECK-LABEL: <__Thumbv4ABSLongThunk_far2>:

//--- a.s
 .thumb
 .global _start
 .type _start, %function
 .section .text.00, "ax", %progbits

_start:
 bl far

 .section .text.over.01, "ax", %progbits
 .global over1
 .type over1, %function
over1:
/// Expect reuse of non-overlay .text.00 thunk.
 bl far
/// Expect generation of one thunk for Overlay.
 bl far2
 bl far2

 .section .text.over.02, "ax", %progbits
 .global over2
 .type over2, %function
over2:
/// Expect reuse of non-overlay .text.00 thunk.
 bl far
/// Expect generation of one thunk for Overlay.
 bl far2
 bl far2
/// Add gap so we can distinguish the thunk by address.
 nop
 nop

 .section .text.02, "ax", %progbits
 .global nonover
 .type nonover, %function
/// Expect another thunk for far2 as we cannot reuse one in an overlay.
nonover:
 bl far2


 .section .text.far, "ax", %progbits
 .global far
 .type far, %function
 .global far2
 .type far2 %function
far:	bx lr
far2:	bx lr

//--- overlay.ld

SECTIONS {
  .text.01 0x1000 : { *(.text.00) }
  OVERLAY 0x2000 : {
    .text.over.01   { *(.text.over.01) }
    .text.over.02   { *(.text.over.02) }
  }
  .text.02 0x3000 : { *(.text.02) }
  .text.03 0x80000000 : { *(.text.far) }
}
