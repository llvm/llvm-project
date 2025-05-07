

// RUN: %clang_cc1 -fbounds-safety -verify %s
// RUN: %clang_cc1 -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc -verify %s

#include <ptrcheck.h>

void Foo(int *__counted_by(len) buf, int len) {}

unsigned long get_len(void *__bidi_indexable ptr);
unsigned long trap_if_bigger_than_max(unsigned long len);


int Test() {
    int arr[10];
    Foo(arr, trap_if_bigger_than_max(get_len(arr)));
    unsigned long len = trap_if_bigger_than_max(get_len(arr));
    Foo(arr, len);
    return 0;
}

// expected-no-diagnostics
