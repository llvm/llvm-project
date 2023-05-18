// RUN: %clang_cc1 -std=c++11 -triple x86_64-apple-darwin10 -emit-llvm -o - %s | FileCheck %s

template<typename T> using id = T;
struct S {
  template<typename T, int N>
    operator id<T[N]>&();
  template<typename T, typename U>
    operator id<T (U::*)()>() const;
};

void f() {
  int (&a)[42] = S(); // CHECK: @_ZN1ScvRAT0__T_IiLi42EEEv(
  char (S::*fp)() = S(); // CHECK: @_ZNK1ScvMT0_FT_vEIcS_EEv(
};
