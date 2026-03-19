// REQUIRES: aarch64-registered-target
// RUN: %clang -cc1as -triple aarch64 %s -filetype obj -mmapsyms=implicit -o %t.o
// RUN: llvm-readelf -s %t.o | FileCheck %s

// CHECK: Symbol table '.symtab' contains 1 entries:
nop

.data
.quad 0
