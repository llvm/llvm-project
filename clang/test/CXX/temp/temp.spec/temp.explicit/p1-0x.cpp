// RUN: %clang_cc1 -fsyntax-only -std=c++11 -verify %s

template<typename T>
struct X {
  void f() {}
};

template inline void X<int>::f(); // expected-error{{explicit instantiation cannot be 'inline'}}

template<typename T>
struct Y {
  constexpr int f() { return 0; }
};

template constexpr int Y<int>::f(); // expected-error{{explicit instantiation cannot be 'constexpr'}}

template<typename T>
struct Z {
  enum E : T { e1, e2 };
  T t; // expected-note {{refers here}}
};

template enum Z<int>::E; // expected-error {{enumerations cannot be explicitly instantiated}}
template int Z<int>::t; // expected-error {{explicit instantiation of 't' does not refer to}}
