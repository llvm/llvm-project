// RUN: %clang_cc1 -std=c++98 -triple x86_64-unknown-unknown %s -verify -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -std=c++11 -triple x86_64-unknown-unknown %s -verify -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -std=c++14 -triple x86_64-unknown-unknown %s -verify -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -std=c++1z -triple x86_64-unknown-unknown %s -verify -fexceptions -fcxx-exceptions -pedantic-errors

#if __cplusplus < 201103L
// expected-error@+1 {{variadic macro}}
#define static_assert(...) __extension__ _Static_assert(__VA_ARGS__)
#endif

namespace dr1813 { // dr1813: 7
  struct B { int i; };
  struct C : B {};
  struct D : C {};
  struct E : D { char : 4; };

  static_assert(__is_standard_layout(B), "");
  static_assert(__is_standard_layout(C), "");
  static_assert(__is_standard_layout(D), "");
  static_assert(!__is_standard_layout(E), "");

  struct Q {};
  struct S : Q {};
  struct T : Q {};
  struct U : S, T {};

  static_assert(__is_standard_layout(Q), "");
  static_assert(__is_standard_layout(S), "");
  static_assert(__is_standard_layout(T), "");
  static_assert(!__is_standard_layout(U), "");
}

namespace dr1881 { // dr1881: 7
  struct A { int a : 4; };
  struct B : A { int b : 3; };
  static_assert(__is_standard_layout(A), "");
  static_assert(!__is_standard_layout(B), "");

  struct C { int : 0; };
  struct D : C { int : 0; };
  static_assert(__is_standard_layout(C), "");
  static_assert(!__is_standard_layout(D), "");
}

void dr1891() { // dr1891: 4
#if __cplusplus >= 201103L
  int n;
  auto a = []{}; // expected-note 2{{candidate}} expected-note 2{{here}}
  auto b = [=]{ return n; }; // expected-note 2{{candidate}} expected-note 2{{here}}
  typedef decltype(a) A;
  typedef decltype(b) B;

  static_assert(!__has_trivial_constructor(A), "");
  static_assert(!__has_trivial_constructor(B), "");

  A x; // expected-error {{no matching constructor}}
  B y; // expected-error {{no matching constructor}}

  a = a; // expected-error {{copy assignment operator is implicitly deleted}}
  a = static_cast<A&&>(a); // expected-error {{copy assignment operator is implicitly deleted}}
  b = b; // expected-error {{copy assignment operator is implicitly deleted}}
  b = static_cast<B&&>(b); // expected-error {{copy assignment operator is implicitly deleted}}
#endif
}
