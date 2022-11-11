// RUN: %clang_cc1 -emit-llvm %s -o - -triple=ppc64-windows-msvc | FileCheck %s

// The purpose of this test is to see that we do something reasonable for
// architectures where we haven't checked what MSVC does.

struct A {
  A() : a(42) {}
  A(const A &o) : a(o.a) {}
  ~A() {}
  int a;
};

struct B {
  A foo(A o);
};

A B::foo(A x) {
  return x;
}

// CHECK-LABEL: define{{.*}} void @"?foo@B@@QEAA?AUA@@U2@@Z"(ptr {{[^,]*}} %this, ptr noalias sret(%struct.A) align 4 %agg.result, ptr noundef %x)
