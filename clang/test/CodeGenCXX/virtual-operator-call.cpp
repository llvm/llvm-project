// RUN: %clang_cc1 %s -triple i386-unknown-unknown -emit-llvm -o - | FileCheck %s

struct A {
  virtual int operator-();
};

void f(A a, A *ap) {
  // CHECK: call noundef i32 @_ZN1AngEv(ptr {{[^,]*}} %a)
  -a;

  // CHECK: call noundef i32 %
  -*ap;
}
