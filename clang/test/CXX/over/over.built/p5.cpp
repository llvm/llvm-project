// RUN: %clang_cc1 -std=c++11 -verify %s -Wno-tautological-compare

void f(int i, bool b) {
  (void)--i;
  (void)i--;

  (void)--b; // expected-error {{cannot decrement expression of type bool}}
  (void)b--; // expected-error {{cannot decrement expression of type bool}}
}

