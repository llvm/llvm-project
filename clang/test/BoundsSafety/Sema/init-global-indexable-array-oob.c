
// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -verify %s
// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc -verify %s

#include <ptrcheck.h>

int array[10] = {0};
int *__indexable idxa = array - 1; // expected-error{{initializer element is not a compile-time constant}}
int *__indexable idxb = array;
int *__indexable idxc = array + 9;
int *__indexable idxd = array + 10; // expected-error{{initializer element is not a compile-time constant}}

struct {
    int p;
    int q[10];
} struct_with_array;
int *__indexable idxe = struct_with_array.q - 1; // expected-error{{initializer element is not a compile-time constant}}
int *__indexable idxf = struct_with_array.q;
int *__indexable idxg = struct_with_array.q + 9;
int *__indexable idxh = struct_with_array.q + 10; // expected-error{{initializer element is not a compile-time constant}}
