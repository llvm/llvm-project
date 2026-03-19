// REQUIRES: aarch64
// RUN: rm -rf %t && split-file %s %t && cd %t
// RUN: llvm-mc -filetype=obj -triple=aarch64 asm -o a.o
// RUN: ld.lld --script=lds a.o -o exe --defsym absolute=0xf0000000
// RUN: llvm-objdump -d --no-show-raw-insn exe | FileCheck %s

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

.section .text.0,"axy",@progbits
.global _start
.type _start,@function
_start:
/// Expect thunk to target a linker generated entry point with BTI landing pad.
/// Two calls to make sure only one landing pad is created.
  bl fn1
  b  fn1
/// No BTI landing pad is added for absolute symbols.
  bl absolute

/// This function does not have a BTI compatible landing pad. Expect a linker
/// generated landing pad for indirect branch thunks.
.section .text.1,"axy",@progbits
.hidden fn1
.type fn1,@function
fn1:
  ret

// CHECK-LABEL: <_start>:
// CHECK-NEXT:  18001000: bl      0x1800100c <__AArch64AbsXOLongThunk_>
// CHECK-NEXT:            b       0x1800100c <__AArch64AbsXOLongThunk_>
// CHECK-NEXT:            bl      0x18001020 <__AArch64AbsXOLongThunk_absolute>

// CHECK-LABEL: <__AArch64AbsXOLongThunk_>:
// CHECK-NEXT:  1800100c: mov     x16, #0x0
// CHECK-NEXT:            movk    x16, #0x3000, lsl #16
// CHECK-NEXT:            movk    x16, #0x0, lsl #32
// CHECK-NEXT:            movk    x16, #0x0, lsl #48
// CHECK-NEXT:            br      x16

// CHECK-LABEL: <__AArch64AbsXOLongThunk_absolute>:
// CHECK-NEXT:  18001020: mov     x16, #0x0
// CHECK-NEXT:            movk    x16, #0xf000, lsl #16
// CHECK-NEXT:            movk    x16, #0x0, lsl #32
// CHECK-NEXT:            movk    x16, #0x0, lsl #48
// CHECK-NEXT:            br      x16

// CHECK-LABEL: <__AArch64BTIThunk_>:
// CHECK-NEXT:  30000000: bti     c

// CHECK-LABEL: <fn1>:
// CHECK-NEXT:  30000004: ret

//--- lds
PHDRS {
  low PT_LOAD FLAGS(0x1 | 0x4);
  mid PT_LOAD FLAGS(0x1 | 0x4);
  high PT_LOAD FLAGS(0x1 | 0x4);
}
SECTIONS {
  .rodata 0x10000000 : { *(.note.gnu.property) } :low
  .text 0x18001000 : { *(.text.0) } :mid
  .text_high 0x30000000 : { *(.text.*) } :high
}
