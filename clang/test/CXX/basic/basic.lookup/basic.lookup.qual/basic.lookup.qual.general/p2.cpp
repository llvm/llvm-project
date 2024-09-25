// RUN: %clang_cc1 -verify -Wno-unused %s

struct A {
  int y;
};

struct B; // expected-note 4{{forward declaration of 'B'}}

void f(A *a, B *b) {
  a->B::x; // expected-error {{incomplete type 'B' named in nested name specifier}}
  a->A::x; // expected-error {{no member named 'x' in 'A'}}
  a->A::y;
  b->B::x; // expected-error {{member access into incomplete type 'B'}}
  b->A::x; // expected-error {{member access into incomplete type 'B'}}
  b->A::y; // expected-error {{member access into incomplete type 'B'}}
}
