// RUN: %clang_cc1 %s -verify -pedantic -fsyntax-only

int constant c[3] = {0};

void foo() {
  c[0] = 1; //expected-error{{read-only variable is not assignable}}
}
