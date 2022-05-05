// RUN: %clang_cc1 %s -std=gnu89 -pedantic -fsyntax-only -verify

int f(int restrict);

void main(void) {} // expected-warning {{return type of 'main' is not 'int'}} expected-note {{change return type to 'int'}}
