// RUN: %clang_cc1 -fsyntax-only -verify %s 
int x = this; // expected-error {{invalid use of 'this' outside of a non-static member function}}

void f() {
  int x = this; // expected-error {{invalid use of 'this' outside of a non-static member function}}
}
