

// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -verify %s
// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc -verify %s
#include <ptrcheck.h>

extern int *end;
extern int *__ended_by(end) start;

int *end2;
extern int *__ended_by(end2) start2;
// expected-error@-1{{pointer with '__ended_by' and the argument of the attribute must be defined in the same translation unit}}

extern int *end3;
int *__ended_by(end3) start3;
// expected-error@-1{{pointer with '__ended_by' and the argument of the attribute must be defined in the same translation unit}}

extern int *end4;
extern int *__ended_by(end4) start4;

void *end_share;
void *__ended_by(end_share) start5;
// expected-note@-1{{previous use is here}}
void *__ended_by(end_share) start6;
// expected-error@-1{{variable 'end_share' referred to by __ended_by variable cannot be used in other dynamic bounds attributes}}

int *const const_end = 0;
int *data_const_end __unsafe_late_const;

int *__ended_by(const_end) start_with_const_end;
void test_global_start_with_const_end() {
  start_with_const_end = 0;
}

int *__ended_by(data_const_end) start_with_data_const_end;
void test_global_start_with_data_const_end() {
  start_with_data_const_end = 0;
}

void test_extern_start_end() {
  start = 0;
  end = 0;
}

void test_const_end(int *__ended_by(const_end) start);
void test_data_const_end(int *__ended_by(data_const_end) start);

void test_redecl(int *__ended_by(end) start, int* end);
void test_redecl(int *__ended_by(end) start, int* end) {}

void *__ended_by(end) test_redecl_return(int *__ended_by(end) start, int* end);
void *__ended_by(end) test_redecl_return(int *__ended_by(end) start, int* end);


// FIXME: weird error messages
// expected-error@+3{{pointer arithmetic on single pointer 'end' is out of bounds; consider adding '__counted_by' to 'end'}}
// expected-note@+2{{pointer 'end' declared here}}
// expected-error@+1{{'__ended_by' attribute requires a pointer type argument}}
void foo_bin_end(int *__ended_by(end+1) start, int* end);
void foo_cptr_end(int *__ended_by(end) start, char* end);
// expected-error@+1{{'__ended_by' attribute requires a pointer type argument}}
void foo_int_end(int *__ended_by(i) start, int i);
void foo_out_start_out_end(int *__ended_by(*out_end) *out_start, int **out_end);
void foo_out_end(int *__ended_by(*out_end) start, int **out_end);
int *__ended_by(end) foo_ret_end(int *end);

// expected-note@+3{{add a count attribute}}
// expected-error@+2{{parameter of array type 'int[]' decays to a __single pointer}}
// expected-error@+1{{'__ended_by' attribute only applies to pointer arguments}}
void foo_end_in_bracket(int buf[__ended_by(end)], int *end);

// expected-error@+1{{invalid argument expression to bounds attribute}}
int *__ended_by((int *)42) invalid_arg_in_ret_proto(void);

// Check function with no prototype. DO NOT put 'void' in the parentheses.
// expected-error@+1{{invalid argument expression to bounds attribute}}
int *__ended_by((int *)42) invalid_arg_in_ret_noproto();

struct S {
    int *end;
    int *__ended_by(end) start;
    int *__ended_by(end-1) start_1; // expected-error{{invalid argument expression to bounds attribute}}
    int *__bidi_indexable bidi_end;
    int *__ended_by(bidi_end) start_with_bidi_end; // expected-error{{end-pointer must be '__single'}}
    int *__ended_by(end) __bidi_indexable start2; // expected-error{{pointer cannot be '__ended_by' and '__bidi_indexable' at the same time}}
    int *__indexable __ended_by(end) start3; // expected-error{{pointer cannot be '__ended_by' and '__indexable' at the same time}}
};

struct T {
    void *p1;
    void *__ended_by(p1) p2;
    void *__ended_by(p2 - 1) p3; // expected-error{{invalid argument expression to bounds attribute}}
};

struct U {
    void *__sized_by(8) p1;
    void *__ended_by(p1 - 1) p2; // expected-error{{invalid argument expression to bounds attribute}}
};

int baz(void) {
    int *end;
    int *__ended_by(end) start;
    return 0;
}

void type_of(int * __ended_by(end) p, int * end) {
    __typeof__(p) p2; // expected-error{{__typeof__ on an expression of type 'int *__single __ended_by(end)' (aka 'int *__single') is not yet supported}}
}
