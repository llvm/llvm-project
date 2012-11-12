// RUN: %clang_cc1 -fsyntax-only -verify %s

struct C {
  virtual C() = 0; // expected-error{{constructor cannot be declared 'virtual'}}
};

void f() {
 C c;
}
