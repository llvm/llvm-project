// RUN: %clang_cc1 -fsyntax-only -verify %s

void f() {
  void *p = malloc(sizeof(int) * 10); // expected-error{{use of undeclared identifier 'malloc'}} expected-note {{maybe try to include <cstdlib>; 'malloc' is defined in <cstdlib>}}
}

int malloc(double);
