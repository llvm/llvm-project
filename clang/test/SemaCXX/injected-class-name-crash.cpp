// RUN: %clang_cc1 -fsyntax-only -verify %s

template <class T>
struct X : public Foo<Bar { // expected-error {{unknown template name 'Foo'}} expected-error {{use of undeclared identifier 'Bar'}} expected-note {{to match this '<'}}
  X();
}; // expected-error {{expected '>'}} expected-error {{expected '{' after base class list}}


template <class T>
X<T>::X() {
}

namespace GH202320 {
  struct S { S f(); };
  struct S::S;
  // expected-error@-1 {{forward declaration of struct cannot have a nested name specifier}}
  S S::f() { return S(); }
} // namespace GH202320
