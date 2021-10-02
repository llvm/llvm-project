// RUN: %clang_cc1 -fsyntax-only -verify %s

class A1 {};
class B1; // expected-note{{'B1' is not defined, but forward declared here; conversion would be valid if it was derived from 'A1'}}
B1 *b1;
A1 *a1 = b1; // expected-error{{cannot initialize a variable of type 'A1 *' with an lvalue of type 'B1 *'}}

template <class C> class A2 {};
template <class C> class B2;
B2<int> *b2;
A2<int> *a2 = b2; // expected-error{{cannot initialize a variable of type 'A2<int> *' with an lvalue of type 'B2<int> *'}}

typedef struct S s;
const s *f();
s *g1() { return f(); } // expected-error{{cannot initialize return object of type 's *' (aka 'S *') with an rvalue of type 'const s *' (aka 'const S *')}}

B1 *g2() { return f(); } // expected-error{{cannot initialize return object of type 'B1 *' with an rvalue of type 'const s *' (aka 'const S *')}}