// RUN: llvm-mc -filetype=obj -triple=i686-unknown-linux %s -o %t.o
// RUN: echo .global __progname > %t2.s
// RUN: llvm-mc -filetype=obj -triple=i686-unknown-linux %t2.s -o %t2.o
// RUN: ld.lld -shared %t2.o -o %t2.so
// RUN: ld.lld -o %t %t.o %t2.so
// RUN: llvm-readobj -dyn-symbols %t | FileCheck %s


// Inputs/progname-ver.so consists of the assembly file
//
// .global bar
// bar:
// .quad __progname
//
// linked into a library  with the version script
//
// VER_1 {
//  global:
//  bar;
// };
//
// We should create it with lld itself once we it supports that.

// RUN: ld.lld -o %t %t.o %p/Inputs/progname-ver.so
// RUN: llvm-readobj -dyn-symbols %t | FileCheck %s

// CHECK:      Name:     __progname@
// CHECK-NEXT: Value:    0x11000
// CHECK-NEXT: Size:     0
// CHECK-NEXT: Binding:  Global (0x1)
// CHECK-NEXT: Type:     None (0x0)
// CHECK-NEXT: Other:    0
// CHECK-NEXT: Section:  .text
// CHECK-NEXT: }

.global _start, __progname
_start:
__progname:
  nop
