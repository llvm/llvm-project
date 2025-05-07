
// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -verify %s
// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc -verify %s
#include <ptrcheck.h>

int foo(int *__counted_by(*out_len)* out_ptr, int *out_len) {
    int arr[10];
    *&*(&*out_ptr) = arr;
    *out_len = 10;
    if (*out_len == 10) {
        *&*out_ptr = &arr[0] + 1; // expected-error{{assignment to 'int *__single __counted_by(*out_len)' (aka 'int *__single') '*out_ptr' requires corresponding assignment to '*out_len'; add self assignment '*out_len = *out_len' if the value has not changed}}
    }
    return 0;
}

