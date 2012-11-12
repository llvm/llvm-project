// RUN: %clang_cc1 -fsyntax-only -verify %s

template<typename T, T Value> struct Constant; // expected-note{{template parameter is declared here}} \
// FIXME: bad location expected-error{{a non-type template parameter cannot have type 'float'}}

Constant<int, 5> *c1;

int x;
float f(int, double);

Constant<int&, x> *c2;
Constant<int*, &x> *c3;
Constant<float (*)(int, double), f> *c4;
Constant<float (*)(int, double), &f> *c5;

Constant<float (*)(int, int), f> *c6; // expected-error{{non-type template argument of type 'float (int, double)' cannot be converted to a value of type 'float (*)(int, int)'}}

Constant<float, 0> *c7; // expected-note{{while substituting}}
