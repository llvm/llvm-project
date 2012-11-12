// RUN: %clang_cc1 -fsyntax-only -verify %s
class X;

// C++ [temp.param]p4
typedef int INT;
enum E { enum1, enum2 };
template<int N> struct A1;
template<INT N, INT M> struct A2;
template<enum E x, E y> struct A3;
template<int &X> struct A4;
template<int *Ptr> struct A5;
template<int (&f)(int, int)> struct A6;
template<int (*fp)(float, double)> struct A7;
template<int X::*pm> struct A8;
template<float (X::*pmf)(float, int)> struct A9;
template<typename T, T x> struct A10;

template<float f> struct A11; // expected-error{{a non-type template parameter cannot have type 'float'}}

template<void *Ptr> struct A12;
template<int (*IncompleteArrayPtr)[]> struct A13;
