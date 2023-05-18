// RUN: %clang_cc1 -fsyntax-only -verify %s
// expected-no-diagnostics
template<int X[10]> struct A;
template<int *X> struct A;
template<int f(float, double)> struct B;
typedef float FLOAT;
template<int (*f)(FLOAT, double)> struct B;
