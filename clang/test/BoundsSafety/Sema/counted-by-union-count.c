// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -verify %s
// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc -verify %s

// Regression test to make sure we don't crash in the presence of IndirectFieldDecls in count expressionns.

#include <ptrcheck.h>

struct A {
    union {
        unsigned len;
        unsigned other;
    };
    // expected-error@+1{{count expression on struct field may only reference other fields of the same struct}}
    int *__counted_by(len) p;
};

int f(struct A *a) {
    int *q = a->p;
    return *q;
}

struct B {
    struct {
        unsigned len;
        unsigned other;
    };
    // expected-error@+1{{count expression on struct field may only reference other fields of the same struct}}
    int *__counted_by(len) p;
};

int g(struct B *b) {
    int *q = b->p;
    return *q;
}
