// RUN: %clang_cc1 -std=c++17 -verify %s
// expected-no-diagnostics

template<typename T> struct A {
  template<typename U> struct B {
    B(...);
  };
  template<typename U> B(U) -> B<U>;
};
A<void>::B b = 123;

using T = decltype(b);
using T = A<void>::B<int>;
