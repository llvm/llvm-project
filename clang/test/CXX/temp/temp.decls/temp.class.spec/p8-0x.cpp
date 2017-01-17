// RUN: %clang_cc1 -std=c++11 -fsyntax-only -verify %s

template<int ...Values> struct X1;

template<int ...Values>  // expected-note {{non-deducible}}
struct X1<0, Values+1 ...>; // expected-error{{contains a template parameter that cannot be deduced}}

template<typename T, int ...Values> struct X2; // expected-note {{here}}
template<int ...Values> struct X2<X1<Values...>, Values+1 ...> {}; // ok (DR1315)
X2<X1<1, 2, 3>, 2, 3, 4> x2; // ok
X2<X1<1, 2, 3>, 2, 3, 4, 5> x3; // expected-error {{undefined template}}
