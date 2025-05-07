
// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -verify %s
// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc -verify %s
#include <ptrcheck.h>

// expected-note@+1{{passing argument to parameter 'fp' here}}
int* foo(int **__unsafe_indexable fp);
// expected-note@+1{{passing argument to parameter 'fp' here}}
int* fooBound(int *__bidi_indexable fp);

typedef int* (*bar_t)(int *__unsafe_indexable*__bidi_indexable);
typedef int* (*baz_t)(int **__single);

void Test () {
    // expected-note@+1{{pointer 'ptrThin' declared here}}
    int *__single ptrThin;
    // expected-error@+1{{initializing 'int *__bidi_indexable*__single' with an expression of incompatible nested pointer type 'int *__single*__bidi_indexable'; use explicit cast to perform this conversion}}
    int *__bidi_indexable *__single ptrBoundPtrThin = &ptrThin;
    int *__single *__single ptrThinPtrThin = &ptrThin; // ok
    int *__indexable *__single ptrArrayPtrThin;

    // expected-error@+1{{passing 'int *__indexable*__single' to parameter of incompatible nested pointer type 'int *__single*__unsafe_indexable'; use explicit cast to perform this conversion}}
    foo(ptrArrayPtrThin);
    foo(ptrThinPtrThin); // ok
    // expected-warning@+1{{passing type 'int *__single' to parameter of type 'int *__bidi_indexable' results in a __bidi_indexable pointer that will trap if a non-zero offset is dereferenced. consider adding '__counted_by' to 'ptrThin'}}
    fooBound(ptrThin);
    fooBound(*ptrBoundPtrThin); // ok

    long intVal = 0xdeadbeef;
    int *__bidi_indexable ptr = (int *)intVal; // expected-error{{initializing 'int *__bidi_indexable' with an expression of incompatible type 'int *__unsafe_indexable' casts away '__unsafe_indexable' qualifier; use '__unsafe_forge_single' or '__unsafe_forge_bidi_indexable' to perform this conversion}}
    // expected-error@+1{{non-pointer to safe pointer conversion is not allowed with -fbounds-safety; use '__unsafe_forge_single' or '__unsafe_forge_bidi_indexable'}}
    ptr = intVal;

    // expected-error@+1{{incompatible integer to pointer conversion initializing 'int *__unsafe_indexable' with an expression of type 'long'}}
    int *__unsafe_indexable ptrUnsafe = intVal;

    // expected-error@+1{{non-pointer to safe pointer conversion is not allowed with -fbounds-safety; use '__unsafe_forge_single' or '__unsafe_forge_bidi_indexable'}}
    int *__single ptrThin2 = intVal;

    int **ptrPtr = (int**)ptr; // ok
    // expected-warning@+1{{incompatible pointer types assigning}}
    ptrPtr = ptr;

    int ***ptrPtrPtr = (int***)ptr; // ok
    // expected-warning@+1{{incompatible pointer types assigning}}
    ptrPtrPtr = ptr;

    // expected-error@+1{{conversion between pointers to functions with incompatible bound attributes}}
    bar_t fp1 = foo;
    // expected-error@+1{{conversion between pointers to functions with incompatible bound attributes}}
    baz_t fp2 = foo;
}
