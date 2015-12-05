// REQUIRES: aarch64
// RUN: llvm-mc -filetype=obj -triple=aarch64-none-freebsd %s -o %t.o
// RUN: not ld.lld -shared %t.o -o %t.so 2>&1 | FileCheck %s
// CHECK: Relocation R_AARCH64_ABS16 cannot be used when making a shared object; recompile with -fPIC.

.data
  .hword foo
