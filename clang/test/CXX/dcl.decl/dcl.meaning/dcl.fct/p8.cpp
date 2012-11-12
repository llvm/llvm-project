// RUN: %clang_cc1 -fsyntax-only -verify %s

struct A { };
A::A (enum { e1 }) {} // expected-error{{can not be defined in a parameter}} \
// expected-error{{out-of-line definition}}
void A::f(enum { e2 }) {} // expected-error{{can not be defined in a parameter}} \
// expected-error{{out-of-line definition}}

enum { e3 } A::g() { } // expected-error{{can not be defined in the result type}} \
// expected-error{{out-of-line definition}}
