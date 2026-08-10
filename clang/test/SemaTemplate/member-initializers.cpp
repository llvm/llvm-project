// RUN: %clang_cc1 -fsyntax-only -verify %s
// expected-no-diagnostics

template<typename T> struct A {
  A() : j(10), i(10) { }
  
  int i;
  int j;
};

template<typename T> struct B : A<T> {
  B() : A<T>() { }
};

