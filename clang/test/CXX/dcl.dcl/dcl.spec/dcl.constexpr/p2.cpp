// RUN: %clang_cc1 -std=c++11 -emit-llvm %s -o - | FileCheck %s

// constexpr functions and constexpr constructors are implicitly inline.
struct S {
  constexpr S(int n);
  constexpr int g();
  int n;
};

constexpr S::S(int n) : n(n) {}

constexpr S f(S s) {
  return s.n * 2;
}

constexpr int S::g() {
  return f(*this).n;
}

// CHECK: define linkonce_odr {{.*}} @_Z1f1S(
// CHECK: define linkonce_odr {{.*}} @_ZN1SC1Ei(
// CHECK: define linkonce_odr {{.*}} @_ZNK1S1gEv(

int g() {
  return f(42).g();
}
