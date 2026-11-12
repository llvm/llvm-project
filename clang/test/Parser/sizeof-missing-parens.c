// RUN: %clang_cc1 -fsyntax-only -verify %s

void Foo(int);

#define Bar(x) Foo(x)

void Baz(void) {
  Foo(sizeof int); // expected-error {{expected parentheses around type name in sizeof expression}}
  Bar(sizeof int); // expected-error {{expected parentheses around type name in sizeof expression}}
}
