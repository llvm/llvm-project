// RUN: %clang_cc1 -fsyntax-only -std=c++98 -verify %s
template<typename T> struct X;
template<int I> struct Y;

X<X<int> > *x1;
X<X<int>> *x2; // expected-warning{{two consecutive right angle brackets without a space is a C++11 extension}}

X<X<X<X<int>> // expected-warning{{two consecutive right angle brackets without a space is a C++11 extension}}
    >> *x3;   // expected-warning{{two consecutive right angle brackets without a space is a C++11 extension}}

Y<(1 >> 2)> *y1;
Y<1 >> 2> *y2; // expected-warning{{use of right-shift operator ('>>') in template argument will require parentheses in C++11}}
