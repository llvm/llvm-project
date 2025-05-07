
// RUN: %clang_cc1 -triple arm64-apple-ios -fptrauth-intrinsics -fsyntax-only -fbounds-safety -verify %s
// RUN: %clang_cc1 -triple arm64-apple-ios -fptrauth-intrinsics -fsyntax-only -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc -verify %s
#include <ptrcheck.h>

int p;
// expected-warning@+1{{incompatible pointer types initializing 'int *__single' with an expression of type 'char *__bidi_indexable'}}
int *q1 = ((char*)&p) + 1; // expected-error{{initializer element is not a compile-time constant}}
int *q2 = &p;
int *q3 = &p + 1; // expected-error{{initializer element is not a compile-time constant}}

int arr[10];
int *q4 = arr;
int *q5 = arr + 9;
int *q6 = arr + 10; // expected-error{{initializer element is not a compile-time constant}}

extern int iarr[];
// expected-warning@+1{{accessing elements of an unannotated incomplete array always fails at runtime}}
int *q7 = iarr; // expected-error{{initializer element is not a compile-time constant}}

extern int iarr_cnt[__counted_by(10)];
int *q8 = iarr_cnt;
int *q9 = &iarr_cnt[9];
int *q10 = iarr_cnt + 10; // expected-error{{initializer element is not a compile-time constant}}

int g_val;
// expected-warning@+1{{array with '__counted_by' and the argument of the attribute should be defined in the same translation unit}}
extern int iarr_cnt_val[__counted_by(g_val)];
int *q11 = iarr_cnt_val; // expected-error{{initializer element is not a compile-time constant}}

char c;
// expected-warning@+1{{incompatible pointer types initializing 'int *__single' with an expression of type 'char *__bidi_indexable'}}
int *q12 = &c; // expected-error{{initializer element is not a compile-time constant}}
