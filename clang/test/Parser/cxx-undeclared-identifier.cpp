// RUN: %clang_cc1 -fsyntax-only -pedantic -verify %s

namespace ImplicitInt {
  static a(4); // expected-error {{requires a type specifier}}
  b(int n); // expected-error {{requires a type specifier}}
  c (*p)[]; // expected-error {{unknown type name 'c'}}
  itn f(char *p, *q); // expected-error {{unknown type name 'itn'}} expected-error {{requires a type specifier}}

  struct S {
    void f();
  };
  S::f() {} // expected-error {{requires a type specifier}}
}

// PR7180
int f(a::b::c); // expected-error {{use of undeclared identifier 'a'}}

class Foo::Bar { // expected-error {{use of undeclared identifier 'Foo'}} \
                 // expected-note {{to match this '{'}} \
                 // expected-error {{expected ';' after class}}
                 // expected-error {{expected '}'}}
