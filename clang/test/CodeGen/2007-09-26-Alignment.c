// RUN: %clang_cc1 -emit-llvm %s -o - | FileCheck %s
extern void p(int *);
int q(void) {
  // CHECK: alloca i32, align 16
  int x __attribute__ ((aligned (16)));
  p(&x);
  return x;
}
