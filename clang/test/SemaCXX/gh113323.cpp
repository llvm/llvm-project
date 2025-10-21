// RUN: %clang_cc1 -fsyntax-only -verify %s
// RUN: %clang_cc1 -fsyntax-only -verify=expected,no-recovery -fno-recovery-ast %s

int a() {} // expected-warning {{non-void function does not return a value}}
constexpr void (*d)() = a; // expected-error {{cannot initialize a variable of type}}
const void *f = __builtin_function_start(d);  // no-recovery-error {{argument must be a function}}
