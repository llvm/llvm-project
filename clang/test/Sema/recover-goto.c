// RUN: %clang_cc1 -fsyntax-only %s -verify

void a() { // expected-note {{to match this '{'}}
  goto A; // expected-error {{use of undeclared label}}
// expected-error {{expected '}'}}
