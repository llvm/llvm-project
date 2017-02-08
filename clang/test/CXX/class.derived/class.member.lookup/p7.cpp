// RUN: %clang_cc1 -verify %s -Wshadow-field

struct A { int n; };  // expected-note {{declared here}}
struct B { float n; };  // expected-note {{declared here}}
struct C : A, B {};
struct D : virtual C {};
struct E : virtual C { char n; }; // expected-warning {{non-static data member 'n' of 'E' shadows member inherited from type 'A'}} expected-warning {{non-static data member 'n' of 'E' shadows member inherited from type 'B'}}
struct F : D, E {} f;
char &k = f.n;
