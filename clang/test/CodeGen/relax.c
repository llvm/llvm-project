// REQUIRES: x86-registered-target
// RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -emit-obj --mrelax-relocations %s -mrelocation-model pic -o %t
// RUN: llvm-readobj -r %t | FileCheck  %s

// CHECK: R_X86_64_REX_GOTPCRELX foo

extern int foo;
int *f(void) {
  return &foo;
}
