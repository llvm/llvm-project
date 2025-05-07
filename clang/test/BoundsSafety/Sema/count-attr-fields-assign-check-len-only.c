
// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -verify %s
// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc -verify %s

#include <ptrcheck.h>

struct S {
    int *__counted_by(l) bp;
    int *bp2 __counted_by(l+1);
    int l;
};

int foo();

int bar (struct S s) {
    int arr[10];
    s.bp = &arr[1];
    s.bp2 = arr;
    s.l = 9;
    // run-time check here

    for (int i = 0; i < s.l; ++i)
        s.bp[i] = i;

    // expected-error@+2{{assignment to 's.l' requires corresponding assignment to 'int *__single __counted_by(l + 1)' (aka 'int *__single') 's.bp2'; add self assignment 's.bp2 = s.bp2' if the value has not changed}}
    // expected-error@+1{{assignment to 's.l' requires corresponding assignment to 'int *__single __counted_by(l)' (aka 'int *__single') 's.bp'; add self assignment 's.bp = s.bp' if the value has not changed}}
    s.l = 11;

    return s.bp2[11] + s.bp[10];
}
