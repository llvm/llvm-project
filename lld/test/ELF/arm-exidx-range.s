// REQUIRES: arm
// Fails for unclear reasons on 32-bit windows
// UNSUPPORTED: system-windows
// RUN: llvm-mc --arm-add-build-attributes --triple=armv7a-linux-gnueabihf -filetype=obj %s -o %t.o
// RUN: echo "SECTIONS { \
// RUN:         . = 0x80000000; \
// RUN:         .text : { *(.text) } \
// RUN:         .vectors 0xffff0000 : AT(0xffff0000) { *(.vectors) } \
// RUN: } " > %t.script
// RUN: ld.lld --script %t.script %t.o -o %t
// RUN: llvm-readobj -x .ARM.exidx %t | FileCheck %s
/// Adapted from Linux kernel linker script failing due to out of range
/// relocation. The .vectors at 0xffff0000 is a common occurrence as the vector
/// table can only be placed at either 0 or 0xffff0000 in older ARM CPUs.
/// In the example the .vectors won't have an exception table so if LLD creates
/// one then we'll get a relocation out of range error. Check that we don't
/// synthesise a table entry or place a sentinel out of range.

/// Expect only .ARM.exidx from _start and sentinel
// CHECK: Hex dump of section '.ARM.exidx':
// CHECK-NEXT: 0x80000000 10000000 01000000 0c000000 01000000
// CHECK-NOT:  0x80000010

 .text
 .global _start
 .type _start, %function
_start:
 .fnstart
 bx lr
 .cantunwind
 .fnend

 .section .vectors, "ax", %progbits
 .global vecs
 .type vecs, %function
vecs:
 bx lr
