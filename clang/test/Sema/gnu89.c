// RUN: %clang_cc1 %s -std=gnu89 -pedantic -fsyntax-only -verify

int f(int restrict);

void main() {} // expected-warning {{return type of 'main' is not 'int'}}
