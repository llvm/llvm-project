// REQUIRES: arm
// RUN: llvm-mc -arm-add-build-attributes -filetype=obj -triple=thumbv7a-none-linux-gnueabi %s -o %t.o
// RUN: ld.lld %t.o -o %t
// RUN: llvm-objdump -d %t --triple=thumbv7a | FileCheck %s
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
// CHECK-NEXT:    200b4:       ff f7 fe ff     bl      #-4
// CHECK: <__Thumbv7ABSLongThunk__start>:
// CHECK-NEXT:    200b8:       ff f7 fc bf     b.w     #-8 <_start>

// CHECK: <__Thumbv7ABSLongThunk__start>:
// CHECK:       10200bc:       40 f2 b5 0c     movw    r12, #181
// CHECK-NEXT:  10200c0:       c0 f2 02 0c     movt    r12, #2
// CHECK-NEXT:  10200c4:       60 47   bx      r12
// CHECK: <foo>:
// CHECK-NEXT:  10200c6:       ff f7 f9 ff     bl      #-14
