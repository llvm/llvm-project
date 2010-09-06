// RUN: %clang_cc1 -fsyntax-only -verify %s

struct A {};
int i = __real__ A(); // expected-error {{invalid type 'A' to __real operator}}
int j = __imag__ A(); // expected-error {{invalid type 'A' to __imag operator}}

