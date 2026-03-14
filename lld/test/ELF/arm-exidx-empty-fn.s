// REQUIRES: arm
// RUN: llvm-mc -filetype=obj -triple=armv7a-none-linux-gnueabi %s -o %t.o
// RUN: ld.lld %t.o -o %t -shared

// RUN: llvm-readelf --unwind %t | FileCheck %s

// Check that any exidx sections for empty functions are discarded.

// CHECK:      Entries [
// CHECK-NEXT:   Entry {
// CHECK-NEXT:     FunctionAddress:
// CHECK-NEXT:     Model: CantUnwind
// CHECK-NEXT:   }
// CHECK-NEXT:   Entry {
// CHECK-NEXT:     FunctionAddress:
// CHECK-NEXT:     Model: CantUnwind
// CHECK-NEXT:   }
// CHECK-NEXT: ]

.section .text.f0,"ax",%progbits
.globl f0
f0:
.fnstart
bx lr
.cantunwind
.fnend

.section .text.f1,"ax",%progbits
.globl f1
f1:
.fnstart
.cantunwind
.fnend

.section .text.f2,"ax",%progbits
.globl f2
f2:
.fnstart
bx lr
.cantunwind
.fnend
