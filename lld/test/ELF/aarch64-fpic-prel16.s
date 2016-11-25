// REQUIRES: aarch64
// RUN: llvm-mc -filetype=obj -triple=aarch64-none-freebsd %s -o %t.o
// RUN: not ld.lld -shared %t.o -o %t.so 2>&1 | FileCheck %s
// CHECK: {{.*}}:(.data+0x0): relocation R_AARCH64_PREL16 cannot be used against shared object; recompile with -fPIC.

.data
  .hword foo - .
