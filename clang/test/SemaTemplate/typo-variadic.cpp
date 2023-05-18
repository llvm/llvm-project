// RUN: %clang_cc1 -fsyntax-only %s -verify
int x = m(s...); // expected-error{{pack expansion does not}} expected-error{{undeclared identifier}}
