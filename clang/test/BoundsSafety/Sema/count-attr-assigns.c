
// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -verify %s
// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc -verify %s

#include <ptrcheck.h>

struct S {
    int *__counted_by(l) bp;
    int *bp2;
    int l;
};

int main() {
    struct S s;

    int arr[10];
    s.bp = arr; // expected-error{{assignment to 'int *__single __counted_by(l)' (aka 'int *__single') 's.bp' requires corresponding assignment to 's.l'; add self assignment 's.l = s.l' if the value has not changed}}

    return 0;
}
