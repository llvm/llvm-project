// RUN: %clang_cc1 -std=c++11 -fsyntax-only -verify %s

template<int ...Values> struct X1;

template<int ...Values> 
struct X1<0, Values+1 ...>; // expected-error{{non-type template argument depends on a template parameter of the partial specialization}}


