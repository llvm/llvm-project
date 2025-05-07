
// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -verify %s
// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc -verify %s
#include <ptrcheck.h>

struct nested {
    int *__counted_by(l) bp;
    int *bp2;
    int l;
};

struct S {
    int l;
    struct nested n1;
    int *__counted_by(l) ptr;
    struct nested n2;
};

int main() {
    struct S s;
    struct S *sp = &s;

    int arr[10];
    sp->n1.bp = arr;
    sp->n1.l = 9;

    sp->n2.l = 10; // expected-error{{assignment to 'sp->n2.l' requires corresponding assignment to 'int *__single __counted_by(l)' (aka 'int *__single') 'sp->n2.bp'; add self assignment 'sp->n2.bp = sp->n2.bp' if the value has not changed}}
    sp->n1.l = 0; // expected-error{{assignment to 'sp->n1.l' requires corresponding assignment to 'int *__single __counted_by(l)' (aka 'int *__single') 'sp->n1.bp'; add self assignment 'sp->n1.bp = sp->n1.bp' if the value has not changed}}
    sp->n2.bp = &arr[1]; // expected-error{{assignment to 'int *__single __counted_by(l)' (aka 'int *__single') 'sp->n2.bp' requires corresponding assignment to 'sp->n2.l'; add self assignment 'sp->n2.l = sp->n2.l' if the value has not changed}}

    return 0;
}
