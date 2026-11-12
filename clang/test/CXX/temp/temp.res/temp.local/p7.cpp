// RUN: %clang_cc1 -fsyntax-only -verify %s
// expected-no-diagnostics

template<class T> struct A { 
  int B;
  int f();
}; 

template<class B> int A<B>::f() {
  return B;
}
