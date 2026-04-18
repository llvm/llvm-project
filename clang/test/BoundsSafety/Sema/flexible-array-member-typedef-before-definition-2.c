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

void sink(struct S *__bidi_indexable p);

void foo(struct S *__single p) { sink(p); }
void bar(S_ptr p) { sink(p); }

