// RUN: %clang_cc1 -fsyntax-only -verify %s

int a() {} // expected-warning {{non-void function does not return a value}}
constexpr void (*d)() = a; // expected-error {{cannot initialize a variable of type}}
const void *f = __builtin_function_start(d);
