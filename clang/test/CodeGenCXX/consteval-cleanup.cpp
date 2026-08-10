// RUN: %clang_cc1 -std=c++20 -Wno-unused-value -triple x86_64-linux-gnu -emit-llvm %s -o - | FileCheck %s

struct P {
  consteval P() {}
};

struct A {
  A(int v) { this->data = new int(v); }
  ~A() { delete data; }
private:
  int *data;
};

void foo() {
  for (;A(1), P(), false;);
  // CHECK: foo
  // CHECK: for.cond:
  // CHECK: call void @_ZN1AC1Ei
  // CHECK: call void @_ZN1AD1Ev
  // CHECK: for.body
}
