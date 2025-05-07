
// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -verify %s
// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc -verify %s

#include <ptrcheck.h>

struct S {
    int *__counted_by(count) ptr;
    int count;
};

void foo(void) {
    int p = 0;
    struct S h = { &p }; // expected-warning{{possibly initializing 'h.ptr' of type 'int *__single __counted_by(count)' (aka 'int *__single') and implicit count value of 0 with non-null, which creates a non-dereferenceable pointer; explicitly set count value to 0 to remove this warning}}
    struct S i = { .ptr = &p }; // expected-warning{{possibly initializing 'i.ptr' of type 'int *__single __counted_by(count)' (aka 'int *__single') and implicit count value of 0 with non-null, which creates a non-dereferenceable pointer; explicitly set count value to 0 to remove this warning}}

    struct S a = { 0, 0 }; // ok
    struct S b = { 0 }; // ok
    struct S c = { }; // ok
    struct S d = { .ptr = 0, .count = 0 }; // ok
    struct S e = { .ptr = 0 }; // ok

    struct S f = { &p, 0 }; // ok
    struct S g = { .ptr = &p, .count = 0 }; // ok
}

struct T {
    int *__counted_by(count * 1) ptr;
    int count;
};

void bar(void) {
    int p = 0;
    struct T h = { &p }; // expected-warning{{possibly initializing 'h.ptr' of type 'int *__single __counted_by(count * 1)' (aka 'int *__single') and implicit count value of 0 with non-null, which creates a non-dereferenceable pointer; explicitly set count value to 0 to remove this warning}}
    struct T i = { .ptr = &p }; // expected-warning{{possibly initializing 'i.ptr' of type 'int *__single __counted_by(count * 1)' (aka 'int *__single') and implicit count value of 0 with non-null, which creates a non-dereferenceable pointer; explicitly set count value to 0 to remove this warning}}

    struct T a = { 0, 0 }; // ok
    struct T b = { 0 }; // ok
    struct T c = { }; // ok
    struct T d = { .ptr = 0, .count = 0 }; // ok
    struct T e = { .ptr = 0 }; // ok

    struct T f = { &p, 0 }; // ok
    struct T g = { .ptr = &p, .count = 0 }; // ok
}
