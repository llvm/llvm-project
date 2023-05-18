// RUN: %clang_cc1 -std=c++11 %s -verify

template<int i> class X { /* ... */ };
X< 1>2 > x1; // expected-error{{expected unqualified-id}}
X<(1>2)> x2; // OK
template<class T> class Y { /* ... */ };
Y<X<1>> x3; // OK, same as Y<X<1> > x3; 
Y<X<6>>1>> x4; // expected-error{{expected unqualified-id}}
Y<X<(6>>1)>> x5;

int a, b;
Y<decltype(a < b)> x6;
