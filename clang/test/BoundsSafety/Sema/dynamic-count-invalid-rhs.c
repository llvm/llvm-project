

// RUN: %clang_cc1 -fsyntax-only -verify -fbounds-safety %s
#include <stddef.h>
#include <ptrcheck.h>

// regression test for crash

struct S {
    int * __counted_by(len) ptr;
    int len;
    int fam[__counted_by(len)];
};

void bar(struct S * __bidi_indexable p, int * __counted_by(len) p2, int len) {
    struct S * __single p3 = p;
    // expected-error@+1{{use of undeclared identifier 'asdf'}}
    p3->len = asdf;
    p3->ptr = p2;
}
