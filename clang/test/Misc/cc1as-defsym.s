// REQUIRES: x86-registered-target
// RUN: %clang -cc1as -triple x86_64 -filetype obj --defsym A=1 %s -o %t.o
// RUN: llvm-nm %t.o | FileCheck %s

// CHECK: 0000000000000001 A foo

.globl foo
.ifdef A
foo = 1
.else
foo = 0
.endif
