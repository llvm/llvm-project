// RUN: %clang_cc1 -fsyntax-only -verify %s

template<typename T=int> struct S {};
template<typename> void f();

void foo(void) {
  foo<<<1;      // expected-error {{expected '>>>'}} expected-note {{to match this '<<<'}}

  foo<<<1,1>>>; // expected-error {{expected '('}}

  foo<<<>>>();  // expected-error {{expected expression}}

  // The following two are parse errors because -std=c++11 is not enabled.

  S<S<S<int>>> s; // expected-error 2{{use '> >'}}
  S<S<S<>>> s1; // expected-error 2{{use '> >'}}
  (void)(&f<S<S<int>>>==0); // expected-error 2{{use '> >'}}
  (void)(&f<S<S<>>>==0); // expected-error 2{{use '> >'}}
}
