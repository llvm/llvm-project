// REQUIRES: arm
// RUN: split-file %s %t
// RUN: llvm-mc -arm-add-build-attributes -filetype=obj -triple=thumbv7a-none-linux-gnueabi %t/test.s -o %t.o
// RUN: ld.lld --script %t/script %t.o -o %t2
// RUN: llvm-objdump --no-print-imm-hex --no-show-raw-insn -d %t2 | FileCheck %s

// RUN: llvm-mc -arm-add-build-attributes -filetype=obj -triple=thumbv7aeb-none-linux-gnueabi -mcpu=cortex-a8 %t/test.s -o %t.o
// RUN: ld.lld --script %t/script %t.o -o %t2
// RUN: llvm-objdump --no-print-imm-hex --no-show-raw-insn -d %t2 | FileCheck %s
// RUN: ld.lld --be8 --script %t/script %t.o -o %t2
// RUN: llvm-objdump --no-print-imm-hex --no-show-raw-insn -d %t2 | FileCheck %s

/// Test that we can reuse thunks between Arm and Thumb callers
/// using a BL. Expect two thunks, one for far, one for far2.

//--- script
SECTIONS {
        .text 0x10000 : { *(.text) }
        .text.far 0x10000000 : AT (0x10000000) { *(.far) }
}

//--- test.s

.syntax unified
.text
.globl _start
.type _start, %function
 .arm
_start:
 bl far
 .thumb
 bl far
 bl far2
 .arm
 bl far2

// CHECK:   00010000 <_start>:
// CHECK-NEXT: 10000: bl      0x10010 <__ARMv7ABSLongThunk_far>
// CHECK-NEXT: 10004: blx     0x10010 <__ARMv7ABSLongThunk_far>
// CHECK-NEXT: 10008: bl      0x1001c <__Thumbv7ABSLongThunk_far2>
// CHECK-NEXT: 1000c: blx     0x1001c <__Thumbv7ABSLongThunk_far2>
// CHECK:   00010010 <__ARMv7ABSLongThunk_far>:
// CHECK-NEXT: 10010: movw    r12, #0
// CHECK-NEXT: 10014: movt    r12, #4096
// CHECK-NEXT: 10018: bx      r12
// CHECK:   0001001c <__Thumbv7ABSLongThunk_far2>:
// CHECK-NEXT: 1001c: movw    r12, #4
// CHECK-NEXT: 10020: movt    r12, #4096
// CHECK-NEXT: 10024: bx      r12

.section .text.far, "ax", %progbits
.globl far
.type far, %function
far:
 bx lr
.globl far2
.type far2, %function
far2:
 bx lr

// CHECK: Disassembly of section .text.far:
// CHECK:      10000000 <far>:
// CHECK-NEXT: 10000000: bx      lr
// CHECK:      10000004 <far2>:
// CHECK-NEXT: 10000004: bx      lr
