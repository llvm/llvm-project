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
    // expected-error@+1{{assignment to 'int *__single __counted_by(len)' (aka 'int *__single') 'p3->ptr' requires corresponding assignment to 'p3->len'; add self assignment 'p3->len = p3->len' if the value has not changed}}
    p3->ptr = p2;
}
