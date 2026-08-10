// RUN: %clang_cc1 -fsyntax-only -verify %s

struct A { };

void f() {
  struct B : private A {}; // expected-note{{declared private here}}
  
  B b;
  
  A *a = &b; // expected-error{{cannot cast 'B' to its private base class 'A'}}
}
