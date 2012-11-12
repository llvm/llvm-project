// RUN: %clang_cc1 -fsyntax-only -verify %s

// PR9137
void f0(int x) : {}; // expected-error{{expected function body after function declarator}}
void f1(int x) try {}; // expected-error{{expected function body after function declarator}}
