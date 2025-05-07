

// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -verify %s
// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc -verify %s

#include <ptrcheck.h>

extern struct a __a;

struct b {
    void *a;
};

struct b b = {
    .a = &__a,
};

void foo(struct b b, struct a * a) {
    a = b.a;
}

struct c;

void bar(struct c * c, struct a * a) {
    a = c; // expected-warning{{incompatible pointer types assigning to 'struct a *__single' from 'struct c *__single'}}
}
