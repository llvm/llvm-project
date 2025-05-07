
// RUN: not %clang_cc1 -fsyntax-only -fbounds-safety -fdiagnostics-parseable-fixits %s 2>&1 | FileCheck %s --implicit-check-not="fix-it"
// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -verify %s
// RUN: not %clang_cc1 -fsyntax-only -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc -fdiagnostics-parseable-fixits %s 2>&1| FileCheck %s --implicit-check-not="fix-it"

#include <ptrcheck.h>

struct my_struct {
    int len;
    int *__counted_by_or_null(len) tmp;
    // CHECK: fix-it:{{.*}}:{[[@LINE+1]]:13-[[@LINE+1]]:33}:"__counted_by"
    int fam[__counted_by_or_null(len)]; // expected-error{{flexible array members cannot be null; did you mean __counted_by instead?}}
};

struct constant_sized_inner_arr_2d_struct {
    int len;
    // CHECK: fix-it:{{.*}}:{[[@LINE+1]]:13-[[@LINE+1]]:33}:"__counted_by"
    int fam[__counted_by_or_null(len)][10]; // expected-error{{flexible array members cannot be null; did you mean __counted_by instead?}}
};

// expected-error@+3{{'__counted_by_or_null' attribute on nested pointer type is only allowed on indirect parameters}}
// expected-error@+2{{cannot apply '__counted_by_or_null' attribute to 'int (*)[][size]' because 'int[][size]' has unknown size; did you mean to use '__sized_by_or_null' instead?}}
// CHECK: fix-it:{{.*}}:{[[@LINE+1]]:52-[[@LINE+1]]:72}:"__sized_by_or_null"
void counted_nested_unsized_array(int size, int (* __counted_by_or_null(size) param)[__counted_by_or_null(10)][size]);

void local_array() {
    int local_buf[__counted_by_or_null(29)] = {}; // expected-error{{arrays with an explicit size decay to counted pointers and cannot also have a count attribute}}
    int local_sized_buf[__sized_by_or_null(29)] = {}; // expected-error{{arrays with an explicit size decay to counted pointers and cannot also have a count attribute}}

    int local_buf_no_init[__counted_by_or_null(31)]; // expected-error{{definition of variable with array type needs an explicit size or an initializer}}
    int local_sized_buf_no_init[__sized_by_or_null(31)]; // expected-error{{definition of variable with array type needs an explicit size or an initializer}}
}

int global_buf[__counted_by_or_null(5)] = {}; // expected-error{{arrays with an explicit size decay to counted pointers and cannot also have a count attribute}}

// CHECK: fix-it:{{.*}}:{[[@LINE+1]]:40-[[@LINE+1]]:60}:"__counted_by"
void counted_pointer_to_array(int (*p)[__counted_by_or_null(len)], int len); // expected-error{{array objects cannot be null; did you mean __counted_by instead}}
// CHECK: fix-it:{{.*}}:{[[@LINE+1]]:38-[[@LINE+1]]:56}:"__sized_by"
void sized_pointer_to_array(int (*p)[__sized_by_or_null(size)], int size); // expected-error{{array objects cannot be null; did you mean __sized_by instead}}

extern int len;
// CHECK: fix-it:{{.*}}:{[[@LINE+1]]:16-[[@LINE+1]]:36}:"__counted_by"
extern int arr[__counted_by_or_null(len)]; // expected-error{{array objects cannot be null; did you mean __counted_by instead}}
extern int arr2[__counted_by(len)];
extern int (*arr3)[__counted_by_or_null(len)]; // expected-error{{'__counted_by_or_null' attribute on nested pointer type is only allowed on indirect parameters}}

void local_extern() {
    extern int len2;
    extern int arr4[__counted_by(len2)];
    // CHECK: fix-it:{{.*}}:{[[@LINE+1]]:21-[[@LINE+1]]:41}:"__counted_by"
    extern int arr5[__counted_by_or_null(len2)]; // expected-error{{array objects cannot be null; did you mean __counted_by instead}}
}
