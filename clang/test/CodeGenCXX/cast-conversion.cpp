// RUN: %clang_cc1 -triple x86_64-apple-darwin -std=c++11 -emit-llvm %s -o - | \
// RUN: FileCheck %s
// RUN: %if cir-enabled %{ %clang_cc1 -fclangir -triple x86_64-apple-darwin -std=c++11 -emit-llvm %s -o - | FileCheck %s %}
// RUN: %clang_cc1 -triple i386-apple-darwin -std=c++11 -emit-llvm %s -o - |   \
// RUN: FileCheck %s
// RUN: %if cir-enabled %{ %clang_cc1 -fclangir -triple i386-apple-darwin -std=c++11 -emit-llvm %s -o - | FileCheck %s %}

struct A {
  A(int);
};

struct B {
  B(A);
};

int main () {
  (B)10;
  B(10);
  static_cast<B>(10);
}

// CHECK: call void @_ZN1AC1Ei
// CHECK: call void @_ZN1BC1E1A
// CHECK: call void @_ZN1AC1Ei
// CHECK: call void @_ZN1BC1E1A
// CHECK: call void @_ZN1AC1Ei
// CHECK: call void @_ZN1BC1E1A
