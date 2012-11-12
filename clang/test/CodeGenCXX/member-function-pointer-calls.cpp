// RUN: %clang_cc1 %s -triple=x86_64-apple-darwin10 -emit-llvm -O3 -o - | FileCheck %s
struct A {
  virtual int vf1() { return 1; }
  virtual int vf2() { return 2; }
};

int f(A* a, int (A::*fp)()) {
  return (a->*fp)();
}

// CHECK: define i32 @_Z2g1v()
// CHECK: ret i32 1
int g1() {
  A a;
  return f(&a, &A::vf1);
}

// CHECK: define i32 @_Z2g2v()
// CHECK: ret i32 2
int g2() {
  A a;
  return f(&a, &A::vf2);
}
