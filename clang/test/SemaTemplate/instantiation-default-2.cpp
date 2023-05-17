// RUN: %clang_cc1 -fsyntax-only -verify=expected,precxx17,precxx20 %std_cxx98-14 %s
// RUN: %clang_cc1 -fsyntax-only -verify=expected,cxx17,precxx20 -std=c++17 %s
// RUN: %clang_cc1 -fsyntax-only -verify=expected,cxx20 %std_cxx20- %s

template<typename T, T Value> struct Constant; // precxx17-note{{template parameter is declared here}} \
// FIXME: bad location precxx20-error{{a non-type template parameter cannot have type 'float'}}

Constant<int, 5> *c1;

int x;
float f(int, double);

Constant<int&, x> *c2;
Constant<int*, &x> *c3;
Constant<float (*)(int, double), f> *c4;
Constant<float (*)(int, double), &f> *c5;

Constant<float (*)(int, int), f> *c6; // precxx17-error {{non-type template argument of type 'float (int, double)' cannot be converted to a value of type 'float (*)(int, int)'}} \
                                         cxx17-error {{value of type 'float (int, double)' is not implicitly convertible to 'float (*)(int, int)'}} \
                                         cxx20-error {{value of type 'float (int, double)' is not implicitly convertible to 'float (*)(int, int)'}}

Constant<float, 0> *c7; // precxx20-note {{while substituting}} \
                           cxx20-error {{conversion from 'int' to 'float' is not allowed in a converted constant expression}}
