// RUN: %clang_cc1 -emit-llvm %s -o - | FileCheck %s

unsigned t2(unsigned X) {
  // CHECK: t2
  // CHECK: llvm.ctlz.i32
  return __builtin_clz(X);
}
int t1(int X) {
  // CHECK: t1
  // CHECK: llvm.ctlz.i32
  return __builtin_clz(X);
}
