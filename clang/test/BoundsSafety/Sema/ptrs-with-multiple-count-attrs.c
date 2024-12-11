
// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -verify %s

#define __counted_by(N) __attribute__((__counted_by__(N)))

struct T {
    unsigned n;
    int * end;
    // expected-error@+2{{expected ';' at end of declaration list}}
    // expected-error@+1{{a parameter list without types is only allowed in a function definition}}
    int *__counted_by(3) ended_by(end) ptr;
};