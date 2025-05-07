
// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -verify %s
// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc -verify %s

#include <ptrcheck.h>

struct T {
    int cnt1;
    int cnt2;
    int *__counted_by(2 * cnt1 + 3 * cnt2) ptr;
    int *__counted_by(cnt2) ptr2;
};

void full_group_assigned(void) {
    int arr[10];
    struct T t;

    t.cnt1 = 1;
    t.ptr = arr;
    t.ptr2 = arr;
    t.cnt2 = 2;
}

void missing_cnt1(void) {
    int arr[10];
    struct T t;

    t.ptr = arr; // expected-error{{assignment to 'int *__single __counted_by(2 * cnt1 + 3 * cnt2)' (aka 'int *__single') 't.ptr' requires corresponding assignment to 't.cnt1'; add self assignment 't.cnt1 = t.cnt1' if the value has not changed}}
    t.ptr2 = arr;
    t.cnt2 = 2;
}

void missing_cnt2(void) {
    int arr[10];
    struct T t;

    t.ptr = arr; // expected-error{{assignment to 'int *__single __counted_by(2 * cnt1 + 3 * cnt2)' (aka 'int *__single') 't.ptr' requires corresponding assignment to 't.cnt2'; add self assignment 't.cnt2 = t.cnt2' if the value has not changed}}
    t.cnt1 = 5;
    t.ptr2 = arr;
}

void missing_ptr(void) {
    int arr[10];
    struct T t;

    t.cnt1 = 5; // expected-error{{assignment to 't.cnt1' requires corresponding assignment to 'int *__single __counted_by(2 * cnt1 + 3 * cnt2)' (aka 'int *__single') 't.ptr'; add self assignment 't.ptr = t.ptr' if the value has not changed}}
    t.cnt2 = 0;
    t.ptr2 = arr;
}


void missing_ptr2(void) {
    int arr[10];
    struct T t;

    t.ptr = &arr[0];
    t.cnt1 = 5;
    t.cnt2 = 0; // expected-error{{assignment to 't.cnt2' requires corresponding assignment to 'int *__single __counted_by(cnt2)' (aka 'int *__single') 't.ptr2'; add self assignment 't.ptr2 = t.ptr2' if the value has not changed}}
}

void only_cnt1(void) {
    int arr[10];
    struct T t;

    // expected-error@+1{{assignment to 't.cnt1' requires corresponding assignment to 'int *__single __counted_by(2 * cnt1 + 3 * cnt2)' (aka 'int *__single') 't.ptr'; add self assignment 't.ptr = t.ptr' if the value has not changed}}
    t.cnt1 = 5;
}

void only_ptr(void) {
    int arr[10];
    struct T t;
    // expected-error@+2{{assignment to 'int *__single __counted_by(2 * cnt1 + 3 * cnt2)' (aka 'int *__single') 't.ptr' requires corresponding assignment to 't.cnt1'; add self assignment 't.cnt1 = t.cnt1' if the value has not changed}}
    // expected-error@+1{{assignment to 'int *__single __counted_by(2 * cnt1 + 3 * cnt2)' (aka 'int *__single') 't.ptr' requires corresponding assignment to 't.cnt2'; add self assignment 't.cnt2 = t.cnt2' if the value has not changed}}
    t.ptr = arr;
}

void only_ptr2(void) {
    int arr[10];
    struct T t;
    // expected-error@+1{{assignment to 'int *__single __counted_by(cnt2)' (aka 'int *__single') 't.ptr2' requires corresponding assignment to 't.cnt2'; add self assignment 't.cnt2 = t.cnt2' if the value has not changed}}
    t.ptr2 = arr;
}

void self_assign_ptr_cnt2(void) {
    int arr[10];
    struct T t;

    t.ptr = t.ptr;
    t.ptr2 = arr;
    t.cnt2 = t.cnt2;
    t.cnt1 = 3;
}
