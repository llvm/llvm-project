// REQUIRES: x86

// RUN: llvm-mc -filetype=obj -triple=i386-pc-linux %s -o %t.o
// RUN: ld.lld -shared %t.o -o %t.so
// RUN: llvm-objdump -s %t.so | FileCheck %s

// CHECK:      Contents of section .text:
// CHECK-NEXT: 1000 00000000 00000000

// CHECK: Contents of section .bar:
// CHECK-NEXT:  0000 00100000 fc0f0000

foo:
.quad 0

.section .bar
.long foo - .
.long foo - .
