// RUN: %clang_cc1 -fsyntax-only -verify %s

void foo(void) {
  {
    typedef float X;
  }
  X Y;  // expected-error {{use of undeclared identifier}}
}
