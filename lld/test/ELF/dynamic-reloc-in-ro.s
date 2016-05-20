// REQUIRES: x86
// RUN: llvm-mc -filetype=obj -triple=x86_64-pc-linux %s -o %t.o
// RUN: not ld.lld %t.o -o %t.so -shared 2>&1 | FileCheck %s

foo:
.quad foo

// CHECK: relocation R_X86_64_64 cannot be used when making a shared object; recompile with -fPIC.
