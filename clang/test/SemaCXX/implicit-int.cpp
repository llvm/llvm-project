// RUN: %clang_cc1 -fsyntax-only -verify %s

x; // expected-error{{a type specifier is required for all declarations}}

f(int y) { return y; } // expected-error{{a type specifier is required for all declarations}}
