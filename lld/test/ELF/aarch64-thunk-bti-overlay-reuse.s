// REQUIRES: aarch64
// RUN: rm -rf %t && split-file %s %t && cd %t
// RUN: llvm-mc -filetype=obj -triple=aarch64 a.s -o a.o
// RUN: ld.lld --script=overlay.ld a.o -o overlay --pic-veneer --print-map
// RUN: llvm-objdump -d --no-show-raw-insn overlay | FileCheck %s

/// A range extension thunk in a different overlay should not be shared as we
/// cannot guarantee it is in memory. However some thunks, like the BTI
/// landing pads are logically alterative entry points for functions so will
/// be in memory if the target is in memory.

/// Expect 3 range extension thunks, one per overlay and 1 for the .text
/// section following. However we only want 1 BTI landing pad thunk at
/// the destination.

CHECK-LABEL: 0000000000001000 <.text.over.01>:
CHECK-NEXT: 1000: bl 0x1004
CHECK-LABEL: 0000000000001004 <__AArch64ADRPThunk_>:

CHECK-LABEL: 0000000000001000 <.text.over.02>:
CHECK-NEXT: 1000: bl 0x1008
CHECK-LABEL: 0000000000001008 <__AArch64ADRPThunk_>:

CHECK-LABEL: 0000000080000000 <__AArch64BTIThunk_>:
CHECk-NEXT: 80000000: bti     c
CHECK-LABEL: 0000000080000004 <far>:

//--- a.s
 .section ".note.gnu.property", "a"
 .p2align 3
 .long 4
 .long 0x10
 .long 0x5
 .asciz "GNU"

/// Enable BTI.
 .long 0xc0000000 // GNU_PROPERTY_AARCH64_FEATURE_1_AND.
 .long 4
 .long 1          // GNU_PROPERTY_AARCH64_FEATURE_1_BTI.
 .long 0

 .section .text.over.01, "ax", %progbits
 bl far

 .section .text.over.02, "ax", %progbits
 bl far
 // So thunk in overlay can be distinguished by address.
 nop

 .global _start
 .section .text
_start:
 bl far

 .section .text.far, "ax", %progbits
far:
 ret

//--- overlay.ld

SECTIONS {
  OVERLAY 0x1000 : {
    .text.over.01   { *(.text.over.01) }
    .text.over.02   { *(.text.over.02) }
  }
	.text 0x2000 : { *(.text) }
  OVERLAY 0x80000000 : {
    .text.far { *(.text.far) }
  }
}
