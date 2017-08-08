// RUN: %clang_cc1 -verify -std=c++1z %s

namespace Explicit {
  // Each notional constructor is explicit if the function or function template
  // was generated from a constructor or deduction-guide that was declared explicit.
  template<typename T> struct A {
    A(T);
    A(T*);
  };
  template<typename T> A(T) -> A<T>;
  template<typename T> explicit A(T*) -> A<T>; // expected-note {{explicit}}

  int *p;
  A a(p);
  A b = p;
  A c{p};
  A d = {p}; // expected-error {{selected an explicit deduction guide}}

  using X = A<int>;
  using Y = A<int*>;

  using X = decltype(a);
  using Y = decltype(b);
  using X = decltype(c);
}
