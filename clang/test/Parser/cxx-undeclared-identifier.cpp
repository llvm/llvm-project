// RUN: %clang_cc1 -fsyntax-only -pedantic -verify %s

namespace ImplicitInt {
  static a(4); // expected-error {{a type specifier is required}}
  b(int n); // expected-error {{a type specifier is required}}
  c (*p)[]; // expected-error {{unknown type name 'c'}}
  itn f(char *p, *q); // expected-error {{unknown type name 'itn'}} expected-error {{a type specifier is required}}

  struct S {
    void f();
  };
  S::f() {} // expected-error {{a type specifier is required}}
}

// PR7180
int f(a::b::c); // expected-error {{use of undeclared identifier 'a'}}

class Foo::Bar { // expected-error {{use of undeclared identifier 'Foo'}}
                 // expected-error {{expected unqualified-id}}
