// RUN: %clang_cc1 -no-opaque-pointers %s -triple i386-unknown-unknown -emit-llvm -o - | FileCheck %s

struct A {
  virtual int operator-();
};

void f(A a, A *ap) {
  // CHECK: call noundef i32 @_ZN1AngEv(%struct.A* {{[^,]*}} %a)
  -a;

  // CHECK: call noundef i32 %
  -*ap;
}
