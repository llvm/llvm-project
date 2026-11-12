// RUN: %clang_cc1 -fsyntax-only -verify -Wattributes %s

int x __attribute__((foobar)); // expected-warning {{unknown attribute 'foobar' ignored}}
void z(void) __attribute__((bogusattr)); // expected-warning {{unknown attribute 'bogusattr' ignored}}
