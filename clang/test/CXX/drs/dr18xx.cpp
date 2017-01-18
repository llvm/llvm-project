// RUN: %clang_cc1 -std=c++98 -triple x86_64-unknown-unknown %s -verify -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -std=c++11 -triple x86_64-unknown-unknown %s -verify -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -std=c++14 -triple x86_64-unknown-unknown %s -verify -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -std=c++1z -triple x86_64-unknown-unknown %s -verify -fexceptions -fcxx-exceptions -pedantic-errors

#if __cplusplus < 201103L
// expected-no-diagnostics
#endif

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
