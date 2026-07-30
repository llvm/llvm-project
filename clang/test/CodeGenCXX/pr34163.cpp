// RUN: %clang_cc1 -emit-llvm -debug-info-kind=standalone -triple x86_64-linux-gnu -o - -x c++ %s | FileCheck %s

void f(struct X *) {}

// CHECK: @_ZTV1X =
struct X {
  void a() { delete this; }
  virtual ~X() {}
  virtual void key_function();
};

// CHECK: define {{.*}} @_ZN1X12key_functionEv(
void X::key_function() {}
