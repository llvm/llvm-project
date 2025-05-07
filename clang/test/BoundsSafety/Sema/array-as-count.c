
// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -verify %s
// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc -verify %s

#include <ptrcheck.h>

void ok_counted_by(int buf[__counted_by(count)], int count);
void ok_sized_by(int buf[__sized_by(size)], int size);

void ok_constant_array(void) {
    int foo[4];
}

void ok_variable_array(void) {
    int count = 4;
    int foo[count];
}


// this is OK: `buf` becomes indexable
__ptrcheck_abi_assume_indexable()
void ok_abi_indexable(int buf[]);


// this is not OK: `buf` becomes single
__ptrcheck_abi_assume_single()
typedef int incomplete_int_array_t[];

void fail_typedef_decay_to_single(incomplete_int_array_t array); // expected-error{{parameter of array type 'incomplete_int_array_t' (aka 'int[]') decays to a __single pointer, and will not allow arithmetic}} \
                                                                    expected-note{{add a count attribute within the declarator brackets or convert the parameter to a pointer with a count or size attribute}}

void fail_decay_to_single(int buf[]); // expected-error{{parameter of array type 'int[]' decays to a __single pointer, and will not allow arithmetic}} \
                                         expected-note{{add a count attribute within the declarator brackets or convert the parameter to a pointer with a count or size attribute}}

void fail_fixed_size(int buf[__counted_by(count) 4], int count); // expected-error{{arrays with an explicit size decay to counted pointers and cannot also have a count attribute}}
void fail_variable_size(int count, int buf[__counted_by(count) count]); // expected-error{{arrays with an explicit size decay to counted pointers and cannot also have a count attribute}}

void fail_local_array_constant(void) {
    int foo[__counted_by(4)] = {1, 2, 3, 4}; // expected-error{{arrays with an explicit size decay to counted pointers and cannot also have a count attribute}}
}

void fail_local_array_variable(void) {
    int count = 4;
    int foo[__counted_by(4) count]; // expected-error{{arrays with an explicit size decay to counted pointers and cannot also have a count attribute}}
}

struct fail_struct_fixed_size {
    int count;
    int elems[__counted_by(count) 5]; // expected-error{{arrays with an explicit size decay to counted pointers and cannot also have a count attribute}}
};

struct fail_struct_sized_by {
    int count;
    int elems[__sized_by(count)]; // expected-error{{'__sized_by' cannot apply to arrays: use 'counted_by' instead}}
};

// It's not possible to create a struct field with a variable-length array
// inside with clang, so this case isn't tested. (GCC lets you do that using a
// local struct with a field sized by a local variable, but Clang has a
// diagnostic that says this will "never be supported".)

// These are OK:
typedef int (__array_decay_discards_count_in_parameters int_array_but_decays_t)[10];
void ok_typedef_count_discarded(int_array_but_decays_t array);
void ok_count_discarded(int (__array_decay_discards_count_in_parameters array)[5]);
void ok_count_discarded_count_attr(int_array_but_decays_t __counted_by(count) array, int count);
