// RUN: %clang_cc1 -fsyntax-only %s -verify -Wno-deprecated-non-prototype -std=c99

int x(a) int a; {return a;}
int y(b) int b; {return a;} // expected-error {{use of undeclared identifier}}

// PR2332
int a(a)int a;{a=10;return a;}
