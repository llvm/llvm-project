// RUN: %clang_cc1 -std=c++17 -verify %s
// expected-no-diagnostics

template<typename T> struct A {
  template<typename U> struct B {
    B(...);
    B(const B &) = default;
  };
  template<typename U> B(U) -> B<U>;
};
A<void>::B b = 123;
A<void>::B copy = b;

using T = decltype(b);
using T = A<void>::B<int>;

using Copy = decltype(copy);
using Copy = A<void>::B<int>;
