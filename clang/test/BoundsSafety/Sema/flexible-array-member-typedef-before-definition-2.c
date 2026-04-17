// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -verify %s
// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc -verify %s

#include <ptrcheck.h>

struct S;
typedef struct S *S_ptr;

struct S {
    int count;
    int fam[__counted_by(count)];
};

void sink(struct S *__bidi_indexable p);
// expected-note@-1{{passing argument to parameter 'p' here}}

void foo(struct S *__single p) { sink(p); }
void bar(S_ptr p) { sink(p); }
// expected-warning@-1{{passing type 'struct S *__single' to parameter of type 'struct S *__bidi_indexable' results in a __bidi_indexable pointer that will trap if a non-zero offset is dereferenced. consider adding '__counted_by' to 'p'}}
// expected-note@-2{{pointer 'p' declared here}}

