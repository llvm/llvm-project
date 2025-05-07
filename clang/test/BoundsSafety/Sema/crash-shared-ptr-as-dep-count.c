
// RUN: %clang_cc1 -triple arm64-apple-ios -fptrauth-intrinsics -fsyntax-only -fbounds-safety -verify %s
// RUN: %clang_cc1 -triple arm64-apple-ios -fptrauth-intrinsics -fsyntax-only -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc -verify %s

#include <ptrcheck.h>

void foo(void) {
    int n;
    int* n_ptr1 = &n;
    int* n_ptr2 = &n;
    int *__counted_by(*n_ptr1) local_buf1; // expected-error{{dereference operator in '__counted_by' is only allowed for function parameters}}
    int *__counted_by(*n_ptr2) local_buf2; // expected-error{{dereference operator in '__counted_by' is only allowed for function parameters}}
}
