// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -verify %s
// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc -verify %s

// expected-no-diagnostics

#include <ptrcheck.h>

struct S;
typedef struct S *S_ptr;

struct S {
    int count;
    int fam[__counted_by(count)];
};

void sink(struct S *__single p);

// XFAIL: *
// FIXME: crash/assert
void test(S_ptr __bidi_indexable p) {
    sink(p);
}
