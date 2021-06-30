// REQUIRES: arm
// RUN: llvm-mc -arm-add-build-attributes -filetype=obj -triple=armv7a-none-linux-gnueabi %s -o %t
// RUN: echo "SECTIONS {" > %t.script
// RUN: echo "          .text_low 0x130 : { *(.text) }" >> %t.script
// RUN: echo "          .text_high 0xf0000000 : AT(0x1000) { *(.text_high) }" >> %t.script
// RUN: echo "       } " >> %t.script
// RUN: ld.lld --script %t.script --pie --static %t -o %t2
// RUN: llvm-objdump -d --triple=armv7a-none-linux-gnueabi --no-show-raw-insn %t2 | FileCheck %s

// RUN: llvm-mc -arm-add-build-attributes -filetype=obj -triple=thumbv7a-none-linux-gnueabi %s -o %t3
// RUN: ld.lld --script %t.script --pie %t3 -o %t4
// RUN: llvm-objdump -d --triple=thumbv7a-none-linux-gnueabi --no-show-raw-insn %t4 | FileCheck --check-prefix=CHECK-THUMB %s

// Check that we can create Arm and Thumb v7a Position Independent Thunks that
// can span the address space without triggering overflow errors. We use an
// AT(0x1000) for .text_high to avoid creating an almost 4Gb size file.
 .syntax unified
 .text
 .global _start
 .type _start, %function
_start:
 bl high
 bx lr

 .section .text_high, "ax", %progbits
 .global high
 .type high, %function
high:
 bl _start
 bx lr

// ARMv7a instructions and relocations.

// CHECK: Disassembly of section .text_low:
// CHECK-EMPTY:
// CHECK-NEXT: <_start>:
// CHECK-NEXT:      130:       bl      0x138 <__ARMV7PILongThunk_high>
// CHECK-NEXT:      134:       bx      lr

// CHECK: <__ARMV7PILongThunk_high>:
// CHECK-NEXT:      138:       movw    r12, #65208
// CHECK-NEXT:      13c:       movt    r12, #61439
// 0x140 + 0xEFFF0000 + 0x0000FEB8 + 8 = 0xf0000000 = high
// CHECK-NEXT:      140:       add     r12, r12, pc
// CHECK-NEXT:      144:       bx      r12

// CHECK: Disassembly of section .text_high:
// CHECK-EMPTY:
// CHECK-NEXT: <high>:
// CHECK-NEXT: f0000000:       bl      0xf0000008 <__ARMV7PILongThunk__start>
// CHECK-NEXT: f0000004:       bx      lr

// CHECK: <__ARMV7PILongThunk__start>:
// CHECK-NEXT: f0000008:       movw    r12, #280
// CHECK-NEXT: f000000c:       movt    r12, #4096
// 0xf0000010 + 0x10000000 + 0x0000118 + 8 = bits32(0x100000130),0x130 = _start
// CHECK-NEXT: f0000010:       add     r12, r12, pc
// CHECK-NEXT: f0000014:       bx      r12

// Thumbv7a instructions and relocations
// CHECK-THUMB: Disassembly of section .text_low:
// CHECK-THUMB-EMPTY:
// CHECK-THUMB-NEXT: <_start>:
// CHECK-THUMB-NEXT:      130:       bl      0x138 <__ThumbV7PILongThunk_high>
// CHECK-THUMB-NEXT:      134:       bx      lr

// CHECK-THUMB: <__ThumbV7PILongThunk_high>:
// CHECK-THUMB-NEXT:      138:       movw    r12, #65213
// CHECK-THUMB-NEXT:      13c:       movt    r12, #61439
// 0x140 + 0xEFFF0000 + 0x0000FEBD + 4 = 0xf0000001 = high
// CHECK-THUMB-NEXT:      140:       add     r12, pc
// CHECK-THUMB-NEXT:      142:       bx      r12

// CHECK-THUMB: Disassembly of section .text_high:
// CHECK-THUMB-EMPTY:
// CHECK-THUMB-NEXT: <high>:
// CHECK-THUMB-NEXT: f0000000:       bl      0xf0000008 <__ThumbV7PILongThunk__start>
// CHECK-THUMB-NEXT: f0000004:       bx      lr

// CHECK-THUMB: <__ThumbV7PILongThunk__start>:
// CHECK-THUMB-NEXT: f0000008:       movw    r12, #285
// CHECK-THUMB-NEXT: f000000c:       movt    r12, #4096
// 0xf0000010 + 0x10000000 + 0x000011d +4 = bits32(0x100000131),0x131 = _start
// CHECK-THUMB-NEXT: f0000010:       add     r12, pc
// CHECK-THUMB-NEXT: f0000012:       bx      r12
