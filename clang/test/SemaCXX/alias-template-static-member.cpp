// RUN: %clang_cc1 -fsyntax-only -verify -std=c++17 %s

template <class T>
struct A {
  template <class U>
  using E = U;

  static E u; // expected-error {{declaration of variable 'u' with deduced type 'E' requires an initializer}}
};

decltype(A<int>::u) a;