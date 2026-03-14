// RUN: %clang_cc1 -fsyntax-only -verify %s
struct A
{
  template<int> template<typename T> friend void foo(T) {} // expected-error{{extraneous template parameter list}}
  void bar() { foo(0); } // expected-error{{use of undeclared identifier 'foo'}}
};
