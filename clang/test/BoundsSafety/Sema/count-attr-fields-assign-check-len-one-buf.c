
// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -verify %s
// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc -verify %s

#include <ptrcheck.h>

struct S {
    int *__counted_by(l) bp;
    int *bp2 __counted_by(l+1);
    int l;
};

int foo();

int Test(struct S s) {
    int arr[16] = {0};
    s.bp = &arr[3];
    s.bp2 = &arr[3];
    s.l = 9;
    // run-time check here

    for (int i = 0; i < s.l; ++i)
        s.bp[i] = i;

    s.l = 10; // expected-error{{assignment to 's.l' requires corresponding assignment to 'int *__single __counted_by(l + 1)' (aka 'int *__single') 's.bp2'; add self assignment 's.bp2 = s.bp2' if the value has not changed}}
    s.bp = s.bp;

    return s.bp[9];
}
