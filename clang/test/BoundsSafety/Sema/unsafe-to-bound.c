
// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -verify=expected %s
// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc -verify=expected %s

#include <ptrcheck.h>

int  main() {
    int * __unsafe_indexable up;
    int * __bidi_indexable bp = up; // expected-error{{initializing 'int *__bidi_indexable' with an expression of incompatible type 'int *__unsafe_indexable' casts away '__unsafe_indexable' qualifier}}
    bp = (int *)up; // expected-error{{assigning to 'int *__bidi_indexable' from incompatible type 'int *__unsafe_indexable' casts away '__unsafe_indexable' qualifier}}
    bp = (int *)bp;
    int *__indexable ap = (int *)up; // expected-error{{initializing 'int *__indexable' with an expression of incompatible type 'int *__unsafe_indexable' casts away '__unsafe_indexable' qualifier}}
    ap = up; // expected-error{{assigning to 'int *__indexable' from incompatible type 'int *__unsafe_indexable' casts away '__unsafe_indexable' qualifier}}
    ap = (int *)ap;
    int *auto_p = up; // expected-error{{initializing 'int *__bidi_indexable' with an expression of incompatible type 'int *__unsafe_indexable' casts away '__unsafe_indexable' qualifier}}
    auto_p = up; // expected-error{{assigning to 'int *__bidi_indexable' from incompatible type 'int *__unsafe_indexable' casts away '__unsafe_indexable' qualifier}}
    auto_p = (int*)up; // expected-error{{assigning to 'int *__bidi_indexable' from incompatible type 'int *__unsafe_indexable' casts away '__unsafe_indexable' qualifier}}
    auto_p = (void*__bidi_indexable)0; // no error

    void *vp = up; // expected-error{{initializing 'void *__bidi_indexable' with an expression of incompatible type 'int *__unsafe_indexable' casts away '__unsafe_indexable' qualifier}}
    vp = (void*)0; // no error

    bp = (int*)0;

    typedef int * myintptr_t;
    myintptr_t mp = (myintptr_t)0;
    return 0;
}
