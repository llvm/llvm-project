
// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -verify %s -verify-ignore-unexpected=note
// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc -verify %s -verify-ignore-unexpected=note
#include <ptrcheck.h>
#include "system-header-func-decl.h"

int* fooBound(int *__bidi_indexable fp);

void arrayFoo(int *(*__unsafe_indexable fp)[1]);

typedef int* (*bar_t)(int *__unsafe_indexable*__bidi_indexable);
typedef int* (*baz_t)(int **__single);

void Test () {
    int *__single ptrThin;
    int *__single (*__bidi_indexable ptrThinAPtrBound)[1];

    // initializing

    // expected-error@+1{{initializing 'int *__bidi_indexable*__single' with an expression of incompatible nested pointer type 'int *__single*__bidi_indexable'}}
    int *__bidi_indexable *__single ptrBoundPtrThin = &ptrThin;
    int *__single *__bidi_indexable ptrThinPtrBound = &ptrThin; // ok
    int *__single *__single ptrThinPtrThin = &ptrThin; // ok
    // expected-error@+1{{initializing 'int *__indexable*__single' with an expression of incompatible nested pointer type 'int *__single*__bidi_indexable'}}
    int *__indexable *__single ptrArrayPtrThin = &ptrThin;

    // expected-error@+1{{initializing 'int *__bidi_indexable(*__single)[1]' with an expression of incompatible nested pointer type 'int *__single(*__bidi_indexable)[1]'}}
    int *__bidi_indexable(*__single ptrBoundAPtrThin)[1] = ptrThinAPtrBound;
    int *__single(*__single ptrThinAPtrThin)[1] = ptrThinAPtrBound;
    // expected-error@+1{{initializing 'int *__indexable(*__single)[1]' with an expression of incompatible nested pointer type 'int *__single(*__bidi_indexable)[1]'}}
    int *__indexable(*__single ptrArrayAPtrThin)[1] = ptrThinAPtrBound;

    // expected-error@+1{{initializing 'int *__bidi_indexable*__single' with an expression of incompatible nested pointer type 'int *__single(*__bidi_indexable)[1]'}}
    int *__bidi_indexable *__single ptrBoundPtrThin2 = ptrThinAPtrBound;
    // expected-warning@+1{{incompatible pointer types initializing 'int *__single*__bidi_indexable' with an expression of type 'int *__single(*__bidi_indexable)[1]'}}
    int *__single *__bidi_indexable ptrThinPtrBound2 = ptrThinAPtrBound;
    // expected-warning@+1{{incompatible pointer types initializing 'int *__single*__single' with an expression of type 'int *__single(*__bidi_indexable)[1]'}}
    int *__single *__single ptrThinPtrThin2 = ptrThinAPtrBound;
    // expected-error@+1{{initializing 'int *__indexable*__single' with an expression of incompatible nested pointer type 'int *__single(*__bidi_indexable)[1]'}}
    int *__indexable *__single ptrArrayPtrThin2 = ptrThinAPtrBound;

    // expected-error@+1{{initializing 'int *__bidi_indexable(*__single)[1]' with an expression of incompatible nested pointer type 'int *__single*__bidi_indexable'}}
    int *__bidi_indexable(*__single ptrBoundAPtrThin2)[1] = &ptrThin;
    // expected-warning@+1{{incompatible pointer types initializing 'int *__single(*__bidi_indexable)[1]' with an expression of type 'int *__single*__bidi_indexable'}}
    int *__single(*__bidi_indexable ptrThinAPtrBound2)[1] = &ptrThin;
    // expected-warning@+1{{incompatible pointer types initializing 'int *__single(*__single)[1]' with an expression of type 'int *__single*__bidi_indexable'}}
    int *__single(*__single ptrThinAPtrThin2)[1] = &ptrThin;
    // expected-error@+1{{initializing 'int *__indexable(*__single)[1]' with an expression of incompatible nested pointer type 'int *__single*__bidi_indexable'}}
    int *__indexable(*__single ptrArrayAPtrThin2)[1] = &ptrThin;

    int *__bidi_indexable *__single *__bidi_indexable ptrBoundPtrThinPtrBound = &ptrBoundPtrThin;
    int *__bidi_indexable *__single *__single ptrBoundPtrThinPtrThin = &ptrBoundPtrThin;
    // expected-error@+1{{initializing 'int *__bidi_indexable*__indexable*__bidi_indexable' with an expression of incompatible nested pointer type 'int *__bidi_indexable*__single*__bidi_indexable'}}
    int *__bidi_indexable *__indexable *__bidi_indexable ptrBoundPtrArrayPtrBound = &ptrBoundPtrThin;
    // expected-error@+1{{initializing 'int *__indexable*__single*__bidi_indexable' with an expression of incompatible nested pointer type 'int *__bidi_indexable*__single*__bidi_indexable'}}
    int *__indexable *__single *__bidi_indexable ptrArrayPtrThinPtrBound = &ptrBoundPtrThin;
    // expected-error@+1{{initializing 'int *__indexable*__indexable*__bidi_indexable' with an expression of incompatible nested pointer type 'int *__bidi_indexable*__single*__bidi_indexable'}}
    int *__indexable *__indexable *__bidi_indexable ptrArrayPtrArrayPtrBound = &ptrBoundPtrThin;

    int *__bidi_indexable(*__single * __bidi_indexable ptrBoundAPtrThinPtrBound)[1] = &ptrBoundAPtrThin;
    int *__bidi_indexable(*__single * __single ptrBoundAPtrThinPtrThin)[1] = &ptrBoundAPtrThin;
    // expected-error@+1{{initializing 'int *__bidi_indexable(*__indexable*__bidi_indexable)[1]' with an expression of incompatible nested pointer type 'int *__bidi_indexable(*__single*__bidi_indexable)[1]'}}
    int *__bidi_indexable(*__indexable * __bidi_indexable ptrBoundAPtrArrayPtrBound)[1] = &ptrBoundAPtrThin;
    // expected-error@+1{{initializing 'int *__indexable(*__single*__bidi_indexable)[1]' with an expression of incompatible nested pointer type 'int *__bidi_indexable(*__single*__bidi_indexable)[1]'}}
    int *__indexable(*__single * __bidi_indexable ptrArrayAPtrThinPtrBound)[1] = &ptrBoundAPtrThin;
    // expected-error@+1{{initializing 'int *__indexable(*__indexable*__bidi_indexable)[1]' with an expression of incompatible nested pointer type 'int *__bidi_indexable(*__single*__bidi_indexable)[1]'}}
    int *__indexable(*__indexable * __bidi_indexable ptrArrayAPtrArrayPtrBound)[1] = &ptrBoundAPtrThin;

    // casting

    int *__bidi_indexable *__single ptrBoundPtrThin3 = (int *__bidi_indexable *__single) & ptrThin;
    int *__bidi_indexable *__single implicitPtrBoundPtrThin3 = & ptrThin; // expected-error{{initializing 'int *__bidi_indexable*__single' with an expression of incompatible nested pointer type 'int *__single*__bidi_indexable'}}
    int *__single *__bidi_indexable ptrThinPtrBound3 = (int *__single *__bidi_indexable) & ptrThin;
    int *__single *__bidi_indexable implicitPtrThinPtrBound3 = & ptrThin;
    int *__single *__single ptrThinPtrThin3 = (int *__single *__single) & ptrThin;
    int *__single *__single implicitPtrThinPtrThin3 = & ptrThin;
    int *__indexable *__single ptrArrayPtrThin3 = (int *__indexable *__single) & ptrThin;
    int *__indexable *__single implicitPtrArrayPtrThin3 = & ptrThin; // expected-error{{initializing 'int *__indexable*__single' with an expression of incompatible nested pointer type 'int *__single*__bidi_indexable'}}

    int *__bidi_indexable(*__single ptrBoundAPtrThin3)[1] = (int *__bidi_indexable(*__single)[1])ptrThinAPtrBound;
    int *__bidi_indexable(*__single implicitPtrBoundAPtrThin3)[1] = ptrThinAPtrBound;  // expected-error{{initializing 'int *__bidi_indexable(*__single)[1]' with an expression of incompatible nested pointer type 'int *__single(*__bidi_indexable)[1]'}}
    int *__single(*__bidi_indexable ptrThinAPtrBound3)[1] = (int *__single(*__bidi_indexable)[1])ptrThinAPtrBound;
    int *__single(*__bidi_indexable implicitPtrThinAPtrBound3)[1] = ptrThinAPtrBound;
    int *__single(*__single ptrThinAPtrThin3)[1] = (int *__single(*__single)[1])ptrThinAPtrBound;
    int *__single(*__single implicitPtrThinAPtrThin3)[1] = ptrThinAPtrBound;
    int *__indexable(*__single ptrArrayAPtrThin3)[1] = (int *__indexable(*__single)[1])ptrThinAPtrBound;
    int *__indexable(*__single implicitPtrArrayAPtrThin3)[1] = ptrThinAPtrBound;  // expected-error{{initializing 'int *__indexable(*__single)[1]' with an expression of incompatible nested pointer type 'int *__single(*__bidi_indexable)[1]'}}

    int *__bidi_indexable *__single ptrBoundPtrThin4 = (int *__bidi_indexable *__single)ptrThinAPtrBound;
    int *__bidi_indexable *__single implicitPtrBoundPtrThin4 = ptrThinAPtrBound; // expected-error{{initializing 'int *__bidi_indexable*__single' with an expression of incompatible nested pointer type 'int *__single(*__bidi_indexable)[1]'}}
    int *__single *__bidi_indexable ptrThinPtrBound4 = (int *__single *__bidi_indexable)ptrThinAPtrBound;
    int *__single *__bidi_indexable implicitPtrThinPtrBound4 = ptrThinAPtrBound; // expected-warning{{incompatible pointer types initializing}}
    int *__single *__single ptrThinPtrThin4 = (int *__single *__single)ptrThinAPtrBound;
    int *__single *__single implicitPtrThinPtrThin4 = ptrThinAPtrBound; // expected-warning{{incompatible pointer types initializing}}
    int *__indexable *__single ptrArrayPtrThin4 = (int *__indexable *__single)ptrThinAPtrBound;
    int *__indexable *__single implicitPtrArrayPtrThin4 = ptrThinAPtrBound; // expected-error{{initializing 'int *__indexable*__single' with an expression of incompatible nested pointer type 'int *__single(*__bidi_indexable)[1]'}}

    int *__bidi_indexable(*__single ptrBoundAPtrThin4)[1] = (int *__bidi_indexable(*__single)[1]) & ptrThin;
    int *__bidi_indexable(*__single implicitPtrBoundAPtrThin4)[1] = & ptrThin; // expected-error{{initializing 'int *__bidi_indexable(*__single)[1]' with an expression of incompatible nested pointer type 'int *__single*__bidi_indexable'}}
    int *__single(*__bidi_indexable ptrThinAPtrBound4)[1] = (int *__single(*__bidi_indexable)[1]) & ptrThin;
    int *__single(*__bidi_indexable implicitPtrThinAPtrBound4)[1] = & ptrThin; // expected-warning{{incompatible pointer types initializing}}
    int *__single(*__single ptrThinAPtrThin4)[1] = (int *__single(*__single)[1]) & ptrThin;
    int *__single(*__single implicitPtrThinAPtrThin4)[1] = & ptrThin; // expected-warning{{incompatible pointer types initializing}}
    int *__indexable(*__single ptrArrayAPtrThin4)[1] = (int *__indexable(*__single)[1]) & ptrThin;
    int *__indexable(*__single implicitPtrArrayAPtrThin4)[1] = & ptrThin; // expected-error{{initializing 'int *__indexable(*__single)[1]' with an expression of incompatible nested pointer type 'int *__single*__bidi_indexable'}}

    int *__bidi_indexable *__single *__bidi_indexable ptrBoundPtrThinPtrBound2 = (int *__bidi_indexable *__single *__bidi_indexable) & ptrBoundPtrThin;
    int *__bidi_indexable *__single *__bidi_indexable implicitPtrBoundPtrThinPtrBound2 = & ptrBoundPtrThin;
    int *__bidi_indexable *__single *__single ptrBoundPtrThinPtrThin2 = (int *__bidi_indexable *__single *__single) & ptrBoundPtrThin;
    int *__bidi_indexable *__single *__single implicitPtrBoundPtrThinPtrThin2 = & ptrBoundPtrThin;
    int *__bidi_indexable *__indexable *__bidi_indexable ptrBoundPtrArrayPtrBound2 = (int *__bidi_indexable *__indexable *__bidi_indexable) & ptrBoundPtrThin;
    int *__bidi_indexable *__indexable *__bidi_indexable implicitPtrBoundPtrArrayPtrBound2 = & ptrBoundPtrThin; // expected-error{{initializing 'int *__bidi_indexable*__indexable*__bidi_indexable' with an expression of incompatible nested pointer type 'int *__bidi_indexable*__single*__bidi_indexable'}}
    int *__indexable *__single *__bidi_indexable ptrArrayPtrThinPtrBound2 = (int *__indexable *__single *__bidi_indexable) & ptrBoundPtrThin;
    int *__indexable *__single *__bidi_indexable implicitPtrArrayPtrThinPtrBound2 = & ptrBoundPtrThin; // expected-error{{initializing 'int *__indexable*__single*__bidi_indexable' with an expression of incompatible nested pointer type 'int *__bidi_indexable*__single*__bidi_indexable'}}
    int *__indexable *__indexable *__bidi_indexable ptrArrayPtrArrayPtrBound2 = (int *__indexable *__indexable *__bidi_indexable) & ptrBoundPtrThin;
    int *__indexable *__indexable *__bidi_indexable implicitPtrArrayPtrArrayPtrBound2 = & ptrBoundPtrThin; // expected-error{{initializing 'int *__indexable*__indexable*__bidi_indexable' with an expression of incompatible nested pointer type 'int *__bidi_indexable*__single*__bidi_indexable'}}

    int *__bidi_indexable(*__single * __bidi_indexable ptrBoundAPtrThinPtrBound2)[1] = (int *__bidi_indexable(*__single * __bidi_indexable)[1]) & ptrBoundAPtrThin;
    int *__bidi_indexable(*__single * __bidi_indexable implicitPtrBoundAPtrThinPtrBound2)[1] = & ptrBoundAPtrThin;
    int *__bidi_indexable(*__single * __single ptrBoundAPtrThinPtrThin2)[1] = (int *__bidi_indexable(*__single * __single)[1]) & ptrBoundAPtrThin;
    int *__bidi_indexable(*__single * __single implicitPtrBoundAPtrThinPtrThin2)[1] = & ptrBoundAPtrThin;
    int *__bidi_indexable(*__indexable * __bidi_indexable ptrBoundAPtrArrayPtrBound2)[1] = (int *__bidi_indexable(*__indexable * __bidi_indexable)[1]) & ptrBoundAPtrThin;
    int *__bidi_indexable(*__indexable * __bidi_indexable implicitPtrBoundAPtrArrayPtrBound2)[1] = & ptrBoundAPtrThin; // expected-error{{initializing 'int *__bidi_indexable(*__indexable*__bidi_indexable)[1]' with an expression of incompatible nested pointer type 'int *__bidi_indexable(*__single*__bidi_indexable)[1]'}}
    int *__indexable(*__single * __bidi_indexable ptrArrayAPtrThinPtrBound2)[1] = (int *__indexable(*__single * __bidi_indexable)[1]) & ptrBoundAPtrThin;
    int *__indexable(*__single * __bidi_indexable implicitPtrArrayAPtrThinPtrBound2)[1] = & ptrBoundAPtrThin; // expected-error{{initializing 'int *__indexable(*__single*__bidi_indexable)[1]' with an expression of incompatible nested pointer type 'int *__bidi_indexable(*__single*__bidi_indexable)[1]'}}
    int *__indexable(*__indexable * __bidi_indexable ptrArrayAPtrArrayPtrBound2)[1] = (int *__indexable(*__indexable * __bidi_indexable)[1]) & ptrBoundAPtrThin;
    int *__indexable(*__indexable * __bidi_indexable implicitPtrArrayAPtrArrayPtrBound2)[1] = & ptrBoundAPtrThin; // expected-error{{initializing 'int *__indexable(*__indexable*__bidi_indexable)[1]' with an expression of incompatible nested pointer type 'int *__bidi_indexable(*__single*__bidi_indexable)[1]'}}

    // passing

    // expected-error@+1{{int *__indexable*__single' to parameter of incompatible nested pointer type 'int **'}}
    foo(ptrArrayPtrThin);
    foo(ptrThinPtrBound); // ok
    foo(ptrThinPtrThin); // ok
    // expected-warning@+1{{passing type 'int *__single' to parameter of type 'int *__bidi_indexable' results in a __bidi_indexable pointer that will trap if a non-zero offset is dereferenced}}
    fooBound(ptrThin);
    fooBound(*ptrBoundPtrThin); // ok

    // expected-error@+1{{passing 'int *__indexable(*__single)[1]' to parameter of incompatible nested pointer type 'int *__single(*__unsafe_indexable)[1]'}}
    arrayFoo(ptrArrayAPtrThin);
    arrayFoo(ptrThinAPtrBound); // ok
    arrayFoo(ptrThinAPtrThin);  // ok

    long intVal;
    int *ptr = (int*)intVal; // expected-error{{initializing 'int *__bidi_indexable' with an expression of incompatible type 'int *__unsafe_indexable' casts away '__unsafe_indexable' qualifier; use '__unsafe_forge_single' or '__unsafe_forge_bidi_indexable' to perform this conversion}}
    // expected-error@+1{{non-pointer to safe pointer conversion is not allowed with -fbounds-safety; use '__unsafe_forge_single' or '__unsafe_forge_bidi_indexable'}}
    ptr = intVal;

    // FIXME: rdar://70651808
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
    baz_t fp2 = foo;
}
