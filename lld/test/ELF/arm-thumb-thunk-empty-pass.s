// REQUIRES: arm
// RUN: llvm-mc -arm-add-build-attributes -filetype=obj -triple=thumbv7a-none-linux-gnueabi %s -o %t.o
// RUN: ld.lld %t.o -o %t
// RUN: llvm-objdump -d %t | FileCheck %s
 .syntax unified
 .global _start, foo
 .type _start, %function
 .section .text.start,"ax",%progbits
_start:
 bl _start
 .section .text.dummy1,"ax",%progbits
 .space 0xfffffe
 .section .text.foo,"ax",%progbits
  .type foo, %function
foo:
 bl _start

// CHECK: Disassembly of section .text:
// CHECK-EMPTY:
// CHECK-NEXT: <_start>:
// CHECK-NEXT:    200b4:       f7ff fffe       bl      0x200b4 <_start>
// CHECK: <__Thumbv7ABSLongThunk__start>:
// CHECK-NEXT:    200b8:       f7ff bffc       b.w     0x200b4 <_start>

// CHECK: <__Thumbv7ABSLongThunk__start>:
// CHECK:       10200bc:       f240 0cb5       movw    r12, #181
// CHECK-NEXT:  10200c0:       f2c0 0c02       movt    r12, #2
// CHECK-NEXT:  10200c4:       4760    bx      r12
// CHECK: <foo>:
// CHECK-NEXT:  10200c6:       f7ff fff9       bl      0x10200bc <__Thumbv7ABSLongThunk__start>
