// RUN: %clang_cc1 -fsyntax-only -verify %s

template<class T> class X; 
template<> class X<int>; // expected-note{{forward}}
X<int>* p; 

X<int> x; // expected-error{{incomplete type}}
