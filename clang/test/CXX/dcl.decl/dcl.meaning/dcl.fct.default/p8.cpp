// RUN: %clang_cc1 -fsyntax-only -verify %s
class A { 
  void f(A* p = this) { }	// expected-error{{invalid use of 'this'}}
};
