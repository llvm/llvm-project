// REQUIRES: aarch64
// RUN: split-file %s %t
// RUN: llvm-mc -filetype=obj -triple=aarch64 %t/asm -o %t.o
// RUN: ld.lld --script %t/lds --shared %t.o -o %t.so 2>&1
// RUN: llvm-objdump -d --no-show-raw-insn --print-imm-hex %t.so | FileCheck %s

// Check that Position Independent thunks are generated for shared libraries.

//--- asm
 .section .text_low, "ax", %progbits
 .globl low_target
 .type low_target, %function
low_target:
 // Need thunk to high_target@plt
 bl high_target
 ret
// CHECK: <low_target>:
// CHECK-NEXT:        0:       bl      0x14 <__AArch64ADRPThunk_high_target>
// CHECK-NEXT:                 ret

 .hidden low_target2
 .globl low_target2
 .type low_target2, %function
low_target2:
 // Need thunk to high_target2
 bl high_target2
 // .text_high+8 = high_target2
 bl .text_high+8
 ret
// CHECK: <low_target2>:
// CHECK-NEXT:        8:       bl      0x20 <__AArch64ADRPThunk_high_target2>
// CHECK-NEXT:        c:       bl      0x2c <__AArch64ADRPThunk_>
// CHECK-NEXT:                 ret

// Expect range extension thunks for .text_low
// adrp calculation is (PC + signed immediate) & (!0xfff)
// CHECK: <__AArch64ADRPThunk_high_target>:
// CHECK-NEXT:       14:       adrp    x16, 0x10000000
// CHECK-NEXT:                 add     x16, x16, #0x50
// CHECK-NEXT:                 br      x16
// CHECK: <__AArch64ADRPThunk_high_target2>:
// CHECK-NEXT:       20:       adrp    x16, 0x10000000
// CHECK-NEXT:                 add     x16, x16, #0x8
// CHECK-NEXT:                 br      x16
/// Identical to the previous one, but for the target .text_high+8.
// CHECK: <__AArch64ADRPThunk_>:
// CHECK-NEXT:       2c:       adrp    x16, 0x10000000
// CHECK-NEXT:                 add     x16, x16, #0x8
// CHECK-NEXT:                 br      x16


 .section .text_high, "ax", %progbits
 .globl high_target
 .type high_target, %function
high_target:
 // No thunk needed as we can reach low_target@plt
 bl low_target
 ret
// CHECK: <high_target>:
// CHECK-NEXT: 10000000:       bl 0x10000040 <low_target@plt>
// CHECK-NEXT:                 ret

 .hidden high_target2
 .globl high_target2
 .type high_target2, %function
high_target2:
 // Need thunk to low_target
 bl low_target2
 ret
// CHECK: <high_target2>:
// CHECK-NEXT: 10000008:       bl      0x10000010 <__AArch64ADRPThunk_low_target2>
// CHECK-NEXT:                 ret

// Expect Thunk for .text.high

// CHECK: <__AArch64ADRPThunk_low_target2>:
// CHECK-NEXT: 10000010:       adrp    x16, 0x0
// CHECK-NEXT:                 add     x16, x16, #0x8
// CHECK-NEXT:                 br      x16

// CHECK: Disassembly of section .plt:
// CHECK-EMPTY:
// CHECK-NEXT: <.plt>:
// CHECK-NEXT: 10000020:       stp     x16, x30, [sp, #-0x10]!
// CHECK-NEXT:                 adrp    x16, 0x10000000
// CHECK-NEXT:                 ldr     x17, [x16, #0x1f8]
// CHECK-NEXT:                 add     x16, x16, #0x1f8
// CHECK-NEXT:                 br      x17
// CHECK-NEXT:                 nop
// CHECK-NEXT:                 nop
// CHECK-NEXT:                 nop
// CHECK-EMPTY:
// CHECK-NEXT:   <low_target@plt>:
// CHECK-NEXT: 10000040:       adrp    x16, 0x10000000
// CHECK-NEXT:                 ldr     x17, [x16, #0x200]
// CHECK-NEXT:                 add     x16, x16, #0x200
// CHECK-NEXT:                 br      x17
// CHECK-EMPTY:
// CHECK-NEXT:   <high_target@plt>:
// CHECK-NEXT: 10000050:       adrp    x16, 0x10000000
// CHECK-NEXT:                 ldr     x17, [x16, #0x208]
// CHECK-NEXT:                 add     x16, x16, #0x208
// CHECK-NEXT:                 br      x17

//--- lds
PHDRS {
  low PT_LOAD FLAGS(0x1 | 0x4);
  high PT_LOAD FLAGS(0x1 | 0x4);
}
SECTIONS {
  .text_low : { *(.text_low) } :low
  .text_high 0x10000000 : { *(.text_high) } :high
}
