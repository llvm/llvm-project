// RUN: %clang_cc1 -fsyntax-only -verify -Wmissing-prototypes %s

void f() { } // expected-warning {{no previous prototype for function 'f'}}

// Don't warn about kernel functions.
kernel void g() { }
