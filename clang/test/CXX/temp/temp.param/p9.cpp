// RUN: %clang_cc1 -fsyntax-only -std=c++98 -verify %s

// A default template-argument shall not be specified in a function
// template declaration or a function template definition
template<typename T = int> // expected-warning{{default template arguments for a function template are a C++11 extension}}
  void foo0(T); 
template<typename T = int> // expected-warning{{default template arguments for a function template are a C++11 extension}} 
  void foo1(T) { } 

// [...] nor in the template-parameter-list of the definition of a
// member of a class template.
template<int N>
struct X0 {
  void f();
};

template<int N = 0> // expected-error{{cannot add a default template argument}}
void X0<N>::f() { } 

class X1 {
  template<template<int> class TT = X0> // expected-error{{not permitted on a friend template}}
  friend void f2();
};
