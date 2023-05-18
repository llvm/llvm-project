// RUN: %clang_cc1 -fsyntax-only -verify %s

template <class T> struct AA { 
  template <class C> virtual void g(C); // expected-error{{'virtual' cannot be specified on member function templates}}
  virtual void f();
};
