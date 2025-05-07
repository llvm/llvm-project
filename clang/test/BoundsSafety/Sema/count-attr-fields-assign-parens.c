
// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -verify %s
// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc -verify %s
#include <ptrcheck.h>

struct S {
    int *__counted_by(l) bp;
    int *bp2 __counted_by(l+1);
    int l;
};

int foo (struct S s) {
    int arr[10];
    struct S *sp = &s;
    ((struct S*)sp)->bp = &arr[1];
    ((struct S*)sp)->bp2 = arr;
    ((struct S*)sp)->l = 9;

    if (s.l == 8) {
        // 'bp2' is also associated with 'l' but the compiler complains once at the first assignment of the group
        ((struct S*)sp)->bp = &arr[1]; // expected-error{{assignment to 'int *__single __counted_by(l)' (aka 'int *__single') 'sp->bp' requires corresponding assignment to 'sp->l'; add self assignment 'sp->l = sp->l' if the value has not changed}}
        ((struct S*)sp)->bp2 = arr;
    }

    return s.bp2[8] + s.bp[7];
}
