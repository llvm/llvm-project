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

void sink(int *__bidi_indexable p);

void test_fam_member_access(S_ptr p) {
    sink(p->fam);
}
