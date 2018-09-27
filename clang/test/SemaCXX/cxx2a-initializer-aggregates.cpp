// RUN: %clang_cc1 -std=c++2a %s -verify

namespace class_with_ctor {
  struct A { // expected-note 6{{candidate}}
    A() = default; // expected-note 3{{candidate}}
    int x;
    int y;
  };
  A a = {1, 2}; // expected-error {{no matching constructor}}

  struct B {
    int x;
    int y;
  };
  B b1 = B(); // trigger declaration of implicit ctors
  B b2 = {1, 2}; // ok

  struct C : A {
    A a;
  };
  C c1 = {{}, {}}; // ok, call default ctor twice
  C c2 = {{1, 2}, {3, 4}}; // expected-error 2{{no matching constructor}}
}
