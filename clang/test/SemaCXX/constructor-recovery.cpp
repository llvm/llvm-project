// RUN: %clang_cc1 -fsyntax-only -verify %s

struct C { // expected-note 1+{{candidate}}
  virtual C() = 0; // expected-error{{constructor cannot be declared 'virtual'}}
};

void f() {
 C c; // expected-error {{no matching constructor}}
}
