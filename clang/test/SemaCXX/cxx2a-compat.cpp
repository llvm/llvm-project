// RUN: %clang_cc1 -fsyntax-only -std=c++17 -Wc++2a-compat-pedantic -verify %s
// RUN: %clang_cc1 -fsyntax-only -std=c++2a -pedantic -verify %s

struct A { // expected-note 0+{{candidate}}
  A() = default; // expected-note 0+{{candidate}}
  int x, y;
};
A a1 = {1, 2};
#if __cplusplus <= 201703L
  // expected-warning@-2 {{aggregate initialization of type 'A' with user-declared constructors is incompatible with C++2a}}
#else
  // expected-error@-4 {{no matching constructor}}
#endif
A a2 = {};

struct B : A { A a; };
B b1 = {{}, {}}; // ok
B b2 = {1, 2, 3, 4};
#if __cplusplus <= 201703L
  // expected-warning@-2 2{{aggregate initialization of type 'A' with user-declared constructors is incompatible with C++2a}}
#else
  // expected-error@-4 2{{no viable conversion from 'int' to 'A'}}
#endif
