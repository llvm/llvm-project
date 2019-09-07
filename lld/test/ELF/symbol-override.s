// REQUIRES: x86
// RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %s -o %t1.o
// RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %p/Inputs/symbol-override.s -o %t2.o
// RUN: ld.lld -shared %t2.o -o %t2.so
// RUN: ld.lld %t1.o %t2.so -o %t
// RUN: llvm-nm -D %t | FileCheck %s

// CHECK:      do
// CHECK-NEXT: foo
// CHECK-NOT:  {{.}}

.text
.globl foo
.type foo,@function
foo:
nop

.text
.globl _start
_start:
callq do@plt
