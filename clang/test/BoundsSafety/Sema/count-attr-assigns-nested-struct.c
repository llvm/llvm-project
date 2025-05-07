
// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -verify %s
// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc -verify %s

#include <ptrcheck.h>

struct S {
    int d;
    struct {
        int *__counted_by(l) bp;
        int *bp2;
        int l;
    } nested;
};

int main() {
    struct S s;

    int arr[10];
    s.nested.bp = arr; // expected-error{{assignment to 'int *__single __counted_by(l)' (aka 'int *__single') 's.nested.bp' requires corresponding assignment to 's.nested.l'; add self assignment 's.nested.l = s.nested.l' if the value has not changed}}

    return 0;
}
