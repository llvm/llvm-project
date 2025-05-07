
// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -verify %s
// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc -verify %s
#include <ptrcheck.h>

struct S {
    int haha;
    struct {
        int *__counted_by(l) bp;
        int *bp2 __counted_by(l+1);
        int l;
    } nested;
};

int foo();

int main () {
    int arr[16] = {0};
    // expected-error@+1{{implicitly initializing 's.nested.bp2' of type 'int *__single __counted_by(l + 1)' (aka 'int *__single') and count value of 1 with null always fails}}
    struct S s;
    s.nested.bp = &arr[3];
    s.nested.bp2 = &arr[3];
    s.nested.l = 9;
    // run-time check here

    for (int i = 0; i < s.nested.l; ++i)
        s.nested.bp[i] = i;

    s.nested.l = 10; // expected-error{{assignment to 's.nested.l' requires corresponding assignment to 'int *__single __counted_by(l + 1)' (aka 'int *__single') 's.nested.bp2'; add self assignment 's.nested.bp2 = s.nested.bp2' if the value has not changed}}
    s.nested.bp = s.nested.bp;

    return s.nested.bp[9];
}
