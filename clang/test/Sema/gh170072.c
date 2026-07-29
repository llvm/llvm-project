// RUN: %clang_cc1 -fsyntax-only -verify -Wno-unused %s

int a[-1]; // expected-error {{declared as an array with a negative size}}

void f() {
  extern int a[];
  *a;
}
