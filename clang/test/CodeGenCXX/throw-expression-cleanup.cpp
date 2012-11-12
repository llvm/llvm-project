// RUN: %clang_cc1 %s -triple x86_64-none-linux-gnu -emit-llvm -fcxx-exceptions -fexceptions -std=c++11 -o - | FileCheck %s
// PR13359

struct X {
  ~X();
};
struct Error {
  Error(const X&) noexcept;
};

void f() {
  try {
    throw Error(X());
  } catch (...) { }
}

// CHECK: define void @_Z1fv
// CHECK: call void @_ZN5ErrorC1ERK1X
// CHECK: invoke void @__cxa_throw
// CHECK: landingpad
// CHECK: call void @_ZN1XD1Ev
// CHECK-NOT: __cxa_free_exception
