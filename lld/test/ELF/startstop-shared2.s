// REQUIRES: x86

/// Synthesize __start_* and __stop_* even if there exists a definition in a DSO.

// RUN: echo '.globl __start_foo; __start_foo:' | llvm-mc -filetype=obj -triple=x86_64 - -o %t.o
// RUN: ld.lld -o %t.so -soname=so %t.o -shared
// RUN: llvm-mc -filetype=obj -triple=x86_64 %s -o %t2.o
// RUN: ld.lld -o %t %t2.o %t.so
// RUN: llvm-objdump -s -h %t | FileCheck %s

// CHECK: foo           00000000 0000000000201248

// CHECK: Contents of section .text:
// CHECK-NEXT: 201240 48122000 00000000

.quad __start_foo
.section foo,"ax"
