// RUN: %clang_cc1 -std=c++1z -verify %s
// expected-no-diagnostics

namespace std_example {
  template<typename T, typename U = int> struct S {
    T data;
  };
  template<typename U> S(U) -> S<typename U::type>;

  struct A {
    using type = short;
    operator type();
  };
  S x{A()};
}
