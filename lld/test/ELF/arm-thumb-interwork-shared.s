// REQUIRES: arm
// RUN: llvm-mc -arm-add-build-attributes -filetype=obj -triple=thumbv7a-none-linux-gnueabi %s -o %t
// RUN: ld.lld %t --shared -o %t.so
// RUN: llvm-objdump --no-print-imm-hex -d --no-show-raw-insn %t.so | FileCheck %s
 .syntax unified
 .global sym1
 .global elsewhere
 .weak weakref
sym1:
 b.w elsewhere
 b.w weakref

 bl elsewhere
 bl weakref

// Check that we generate a thunk for an undefined symbol called via a plt
// entry.

// CHECK: Disassembly of section .text:
// CHECK-EMPTY:
// CHECK-NEXT: <sym1>:
// CHECK-NEXT:    101e0: b.w 0x101f0 <__ThumbV7PILongThunk_elsewhere>
// CHECK-NEXT:           b.w 0x101fc <__ThumbV7PILongThunk_weakref>
// CHECK-NEXT:           blx 0x10230
// CHECK-NEXT:           blx 0x10240
// CHECK: <__ThumbV7PILongThunk_elsewhere>:
// CHECK-NEXT:     101f0: movw    r12, #52
// CHECK-NEXT:           movt    r12, #0
// CHECK-NEXT:           add     r12, pc
// CHECK-NEXT:           bx      r12
// CHECK: <__ThumbV7PILongThunk_weakref>:
// CHECK-NEXT:     101fc: movw    r12, #56
// CHECK-NEXT:           movt    r12, #0
// CHECK-NEXT:           add     r12, pc
// CHECK-NEXT:           bx      r12

// CHECK: Disassembly of section .plt:
// CHECK-EMPTY:
// CHECK-NEXT: <.plt>:
// CHECK-NEXT:     10210: str     lr, [sp, #-4]!
// CHECK-NEXT:           add     lr, pc, #0, #12
// CHECK-NEXT:           add     lr, lr, #32
// CHECK-NEXT:           ldr     pc, [lr, #148]!
// CHECK-NEXT:     10220: d4 d4 d4 d4 .word   0xd4d4d4d4
// CHECK-NEXT:           .word   0xd4d4d4d4
// CHECK-NEXT:           .word   0xd4d4d4d4
// CHECK-NEXT:           .word   0xd4d4d4d4
// CHECK-NEXT:     10230: add     r12, pc, #0, #12
// CHECK-NEXT:           add     r12, r12, #32
// CHECK-NEXT:           ldr     pc, [r12, #124]!
// CHECK-NEXT:     1023c: d4 d4 d4 d4 .word   0xd4d4d4d4
// CHECK-NEXT:     10240: add     r12, pc, #0, #12
// CHECK-NEXT:           add     r12, r12, #32
// CHECK-NEXT:           ldr     pc, [r12, #112]!
// CHECK-NEXT:     1024c: d4 d4 d4 d4 .word   0xd4d4d4d4
