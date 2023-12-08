// RUN: %clang_cc1 -fsyntax-only -verify %s

struct A { };
A::A (enum { e1 }) {} // expected-error{{cannot be defined in a parameter}}
void A::f(enum { e2 }) {} // expected-error{{cannot be defined in a parameter}}

enum { e3 } A::g() { } // expected-error{{cannot be defined in the result type}} \
// expected-error{{out-of-line definition}}
