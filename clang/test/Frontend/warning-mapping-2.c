// Check that -w has lower priority than -pedantic-errors.
// RUN: %clang_cc1 -verify -pedantic-errors -w %s

void f0() { f1(); } // expected-error {{implicit declaration of function}}

