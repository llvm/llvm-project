// REQUIRES: aarch64
// RUN: llvm-mc -filetype=obj -triple=aarch64 %s -o %t.o
// RUN: echo "SECTIONS { \
// RUN:       .text_low : { *(.text_low) } \
// RUN:       .text_high 0x10000000 : { *(.text_high) } \
// RUN:       } " > %t.script
// RUN: ld.lld --script %t.script --shared %t.o -o %t.so 2>&1
// RUN: llvm-objdump -d --no-show-raw-insn --print-imm-hex %t.so | FileCheck %s

// Check that Position Independent thunks are generated for shared libraries.
 .section .text_low, "ax", %progbits
 .globl low_target
 .type low_target, %function
low_target:
 // Need thunk to high_target@plt
 bl high_target
 ret
// CHECK: low_target:
// CHECK-NEXT:       d8:       bl      #0x10 <__AArch64ADRPThunk_high_target>
// CHECK-NEXT:                 ret

 .hidden low_target2
 .globl low_target2
 .type low_target2, %function
low_target2:
 // Need thunk to high_target
 bl high_target2
 ret
// CHECK: low_target2:
// CHECK-NEXT:       e0:       bl      #0x14 <__AArch64ADRPThunk_high_target2>
// CHECK-NEXT:                 ret

// Expect range extension thunks for .text_low
// adrp calculation is (PC + signed immediate) & (!0xfff)
// CHECK: __AArch64ADRPThunk_high_target:
// CHECK-NEXT:       e8:       adrp    x16, #0x10000000
// CHECK-NEXT:                 add     x16, x16, #0x40
// CHECK-NEXT:                 br      x16
// CHECK: __AArch64ADRPThunk_high_target2:
// CHECK-NEXT:       f4:       adrp    x16, #0x10000000
// CHECK-NEXT:                 add     x16, x16, #0x8
// CHECK-NEXT:                 br      x16


 .section .text_high, "ax", %progbits
 .globl high_target
 .type high_target, %function
high_target:
 // No thunk needed as we can reach low_target@plt
 bl low_target
 ret
// CHECK: high_target:
// CHECK-NEXT: 10000000:       bl #0x50 <low_target@plt>
// CHECK-NEXT:                 ret

 .hidden high_target2
 .globl high_target2
 .type high_target2, %function
high_target2:
 // Need thunk to low_target
 bl low_target2
 ret
// CHECK: high_target2:
// CHECK-NEXT: 10000008:       bl      #0x8 <__AArch64ADRPThunk_low_target2>
// CHECK-NEXT:                 ret

// Expect Thunk for .text.high

// CHECK: __AArch64ADRPThunk_low_target2:
// CHECK-NEXT: 10000010:       adrp    x16, #-0x10000000
// CHECK-NEXT:                 add     x16, x16, #0xe0
// CHECK-NEXT:                 br      x16

// CHECK: Disassembly of section .plt:
// CHECK-EMPTY:
// CHECK-NEXT: .plt:
// CHECK-NEXT: 10000020:       stp     x16, x30, [sp, #-0x10]!
// CHECK-NEXT:                 adrp    x16, #0
// CHECK-NEXT:                 ldr     x17, [x16, #0x120]
// CHECK-NEXT:                 add     x16, x16, #0x120
// CHECK-NEXT:                 br      x17
// CHECK-NEXT:                 nop
// CHECK-NEXT:                 nop
// CHECK-NEXT:                 nop
// CHECK-EMPTY:
// CHECK-NEXT:   high_target@plt:
// CHECK-NEXT: 10000040:       adrp    x16, #0x0
// CHECK-NEXT:                 ldr     x17, [x16, #0x128]
// CHECK-NEXT:                 add     x16, x16, #0x128
// CHECK-NEXT:                 br      x17
// CHECK-EMPTY:
// CHECK-NEXT:   low_target@plt:
// CHECK-NEXT: 10000050:       adrp    x16, #0x0
// CHECK-NEXT:                 ldr     x17, [x16, #0x130]
// CHECK-NEXT:                 add     x16, x16, #0x130
// CHECK-NEXT:                 br      x17
