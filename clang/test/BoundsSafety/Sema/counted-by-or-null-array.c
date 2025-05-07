
// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -verify %s
// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc -verify %s

#include <ptrcheck.h>

struct my_struct {
    int len;
    int *__counted_by_or_null(len) tmp;
    int fam[__counted_by_or_null(len)]; // expected-error{{flexible array members cannot be null; did you mean __counted_by instead?}}
};

struct struct_2d {
    int len;
    int fam[__counted_by_or_null(len)][__counted_by_or_null(len)]; // expected-error{{array has incomplete element type 'int[]'}}
};

struct sized_2d_struct {
    int len;
    int fam[__sized_by_or_null(len)][__counted_by_or_null(len)]; // expected-error{{array has incomplete element type 'int[]'}}
};

int global_len;
struct variably_sized_inner_arr_2d_struct {
    int len;
    int fam[__counted_by_or_null(len)][global_len]; // expected-error{{fields must have a constant size: 'variable length array in structure' extension will never be supported}}
};

struct constant_sized_inner_arr_2d_struct {
    int len;
    int fam[__counted_by_or_null(len)][10]; // expected-error{{flexible array members cannot be null; did you mean __counted_by instead}}
};

struct constant_sized_outer_arr_2d_struct {
    int len;
    int fam[10][__counted_by_or_null(len)]; // expected-error{{array has incomplete element type 'int[]'}}
};

// expected-error@+1{{'__counted_by_or_null' attribute on nested pointer type is only allowed on indirect parameters}}
void pointers_to_array_params(int size, int (* __sized_by_or_null(size) param)[__counted_by_or_null(10)][size], int len, int arr[__counted_by_or_null(11)][len]);

void counted_unsized_array(int size, int (*param)[__counted_by_or_null(10)][__counted_by_or_null(size)]); // expected-error{{array has incomplete element type 'int[]'}}

// expected-error@+2{{'__counted_by_or_null' attribute on nested pointer type is only allowed on indirect parameters}}
// expected-error@+1{{cannot apply '__counted_by_or_null' attribute to 'int (*)[][size]' because 'int[][size]' has unknown size; did you mean to use '__sized_by_or_null' instead?}}
void counted_nested_unsized_array(int size, int (* __counted_by_or_null(size) param)[__counted_by_or_null(10)][size]);

void counted_decayed_nested(int len, int arr[__counted_by_or_null(11)][__counted_by_or_null(len)]); // expected-error{{array has incomplete element type 'int[]'}}

void counted_decayed_of_variably_sized_array(int len, int arr[__counted_by_or_null(11)][len]);

// expected-warning@+1{{tentative array definition assumed to have one element}}
int global_buf[__counted_by_or_null(5)];
int global_2D_buf[__sized_by(7)][__counted_by_or_null(13)]; // expected-error{{array has incomplete element type 'int[]'}}
int global_2D_buf_const_sized_outer[7][__counted_by_or_null(13)]; // expected-error{{array has incomplete element type 'int[]'}}
