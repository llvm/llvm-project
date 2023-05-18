// RUN: %clang_cc1 -fsyntax-only -verify -Wmissing-prototypes %s

void f() { } // expected-warning {{no previous prototype for function 'f'}}
// expected-note@-1{{declare 'static' if the function is not intended to be used outside of this translation unit}}

// Don't warn about kernel functions.
kernel void g() { }
