// RUN: %clang_cc1 -fsyntax-only -verify %s

struct BaseReturn {};

struct Base {
  virtual BaseReturn Foo() = 0;  // expected-note{{overridden virtual function is here}}
};
struct X {};
struct Derived : Base {
  X Foo();  // expected-error{{virtual function 'Foo' has a different return type ('X') than the function it overrides (which has return type 'BaseReturn')}}
};

Derived d;
