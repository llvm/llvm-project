
// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -verify %s
// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc -verify %s
#include <ptrcheck.h>

int a = -3;

int foo() {
	int buf[10];
    int *__bidi_indexable ptrBound = buf;
    int *__single ptrThin = &ptrBound[1];
    int *__indexable ptrArray = &ptrBound[0];

    ptrBound[9] = 3; // ok

    ptrArray[a] = 5; // run-time error
    ptrArray[2] = 2; // ok
    ptrArray[-1] = 3; // expected-error{{array subscript with a negative index on indexable pointer 'ptrArray' is out of bounds}}

    a[ptrArray] = 5; // run-time error
    2[ptrArray] = -1; // ok
    a[ptrArray] = 5[ptrThin]; // expected-error{{array subscript on single pointer 'ptrThin' must use a constant index of 0 to be in bounds}}
    -3[ptrArray] = 2; // expected-error{{expression is not assignable}}

    ptrThin = ptrArray + (-2); // expected-error{{decremented indexable pointer 'ptrArray' is out of bounds}}
	ptrThin = ptrArray + 4; // ok

    ptrBound = ptrThin + 3; // expected-error{{pointer arithmetic on single pointer 'ptrThin' is out of bounds; consider adding '__counted_by' to 'ptrThin'}}
    // expected-note@-18{{pointer 'ptrThin' declared here}}

    // With casts
    // TODO(dliew): Support suggesting `__counted_by` on the `ptrThin` decl in the presence of this cast.
    char *tmp = ((char *)ptrThin) + 1; // expected-error-re{{pointer arithmetic on single pointer '((char *)ptrThin)' is out of {{bounds$}}}}

    return ptrThin[2]; // expected-error{{array subscript on single pointer 'ptrThin' must use a constant index of 0 to be in bounds}}
}

