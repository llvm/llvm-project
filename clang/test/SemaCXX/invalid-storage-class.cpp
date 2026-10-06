// RUN: %clang_cc1 -fsyntax-only -verify %s

auto class X1 {}; // expected-error {{'auto' is not allowed before a class declaration}}

static struct X2 {}; // expected-error {{'static' is not allowed before a class declaration}}

register union X3 {}; // expected-error {{'register' is not allowed before a class declaration}}