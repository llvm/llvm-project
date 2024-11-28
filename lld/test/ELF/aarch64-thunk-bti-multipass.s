// REQUIRES: aarch64
// RUN: rm -rf %t && split-file %s %t && cd %t
// RUN: llvm-mc -filetype=obj -triple=aarch64 asm -o a.o
// RUN: ld.lld --script=lds a.o -o out
// RUN: llvm-objdump -d --no-show-raw-insn out | FileCheck %s

/// Test that a thunk that at creation time does not need to use a BTI
/// compatible landing pad, but due to other thunk insertion ends up
/// out of short-branch range so a BTI thunk is required after all.

//--- asm
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

.section .text.0, "ax", %progbits
.balign 0x1000
.global _start
.type _start, %function
_start:
/// Call that requires a thunk.
 bl fn1
/// padding so that the thunk for fn1 is placed after this section is
/// sufficiently close to the target to be within short range, but only
/// just so that a small displacement will mean a long thunk is needed.
 .space 0x1000
/// Thunk for call to fn1 will be placed here. Initially it is in short Thunk
/// range of fn1, but due to a thunk added after a later section it won't be
/// and will need a long branch thunk, which in turn needs a BTI landing pad.

// CHECK-LABEL: <_start>:
// CHECK-NEXT: 10001000: bl  0x10002004 <__AArch64AbsLongThunk_fn1>

// CHECK-LABEL: <__AArch64AbsLongThunk_fn1>:
// CHECK-NEXT: 10002004: ldr     x16, 0x1000200c <__AArch64AbsLongThunk_fn1+0x8>
// CHECK-NEXT:           br      x16
// CHECK-NEXT:           00 30 00 18    .word   0x18003000
// CHECK-NEXT:           00 00 00 00    .word   0x00000000

.section .text.1, "ax", %progbits
.balign 0x1000
.global farcall
.type farcall, %function
farcall:
/// Call that requires a thunk.
 bl far
/// Section is aligned to 0x1000 boundary with size multipe of 0x1000.
.space 0x1000 - (. - farcall)
/// Thunk for call to far will be placed here. This will force text.2
/// on to the next alignment boundary, moving it further away from the
/// thunk inserted in the .text_low output section.

// CHECK-LABEL: <farcall>:
// CHECK-NEXT: 18001000: bl      0x18002000 <__AArch64AbsLongThunk_far>

// CHECK-LABEL: <__AArch64AbsLongThunk_far>:
// CHECK-NEXT: 18002000: ldr     x16, 0x18002008 <__AArch64AbsLongThunk_far+0x8>
// CHECK-NEXT:           br      x16
// CHECK-NEXT:           00 00 00 30   .word   0x30000000
// CHECK-NEXT:           00 00 00 00   .word   0x00000000

.section .text.2, "ax", %progbits
.balign 0x1000
.global fn1
.type fn1, %function
fn1:
 ret

.section .text.far, "ax", %progbits
.type far, %function
.global far
far:
 ret

// CHECK-LABEL: <__AArch64BTIThunk_fn1>:
// CHECK-NEXT: 18003000: bti     c
// CHECK-NExT:           b       0x18004000 <fn1>

// CHECK-LABEL: <fn1>:
// CHECK-NEXT: 18004000: ret

// CHECK-LABEL: <__AArch64BTIThunk_far>:
// CHECK-NEXT: 30000000: bti     c

// CHECK-LABEL: <far>:
// CHECK-NEXT: 30000004: ret

//--- lds
PHDRS {
  low PT_LOAD FLAGS(0x1 | 0x4);
  mid PT_LOAD FLAGS(0x1 | 0x4);
  high PT_LOAD FLAGS(0x1 | 0x4);
}
SECTIONS {
  .rodata 0x10000000 : { *(.note.gnu.property) } :low
  .text_low : { *(.text.0) } :low
  .text 0x18001000 : { *(.text.1) } :mid
  .text_aligned : { *(.text.2) } :mid
  .text_high 0x30000000 : { *(.text.far) } :high
}
