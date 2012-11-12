// RUN: %clang_cc1 -std=c++11 -fsyntax-only -verify %s

template<class ...Types> struct Tuple;

Tuple<> *t0;
Tuple<int> *t1;
Tuple<int, char> *t2a;
Tuple<int, float> *t2b = t2a; // expected-error{{cannot initialize a variable of type 'Tuple<int, float> *' with an lvalue of type 'Tuple<int, char> *'}}
Tuple<int, float, double> *t3;
