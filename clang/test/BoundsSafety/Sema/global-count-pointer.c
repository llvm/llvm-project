
// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -verify %s
// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc -verify %s
#include <ptrcheck.h>

// TODO: rdar://85557264
int len = 0;
int *__counted_by(len) ptr;
void *__sized_by(len) ptr_share_len;

const unsigned const_len = 0;
int *__counted_by(const_len) ptr_with_const_len;

int late_const_len __unsafe_late_const = 0;
void *__sized_by(late_const_len) ptr_with_late_const_len;

int arrlen;
// expected-warning@+1{{array with '__counted_by' and the argument of the attribute should be defined in the same translation unit}}
extern int arr[__counted_by(arrlen)];

extern int arrlen2;
extern int arr2[__counted_by(arrlen2)];

extern unsigned extlen;
extern void *__sized_by(extlen) extptr;

extern unsigned extlen2;
// expected-error@+1{{pointer with '__counted_by' and the argument of the attribute must be defined in the same translation unit}}
int *__counted_by(extlen2) ptr2;
// expected-error@+1{{pointer with '__sized_by' and the argument of the attribute must be defined in the same translation unit}}
void *__sized_by(extlen2) ptr3;
// expected-error@+1{{pointer with '__counted_by_or_null' and the argument of the attribute must be defined in the same translation unit}}
int *__counted_by_or_null(extlen2) ptr4;
// expected-error@+1{{pointer with '__sized_by_or_null' and the argument of the attribute must be defined in the same translation unit}}
void *__sized_by_or_null(extlen2) ptr5;

unsigned long len2;
// expected-error@+1{{pointer with '__counted_by' and the argument of the attribute must be defined in the same translation unit}}
extern int *__counted_by(len2) extptr2;
// expected-error@+1{{pointer with '__sized_by' and the argument of the attribute must be defined in the same translation unit}}
extern void *__sized_by(len2) extptr3;
// expected-error@+1{{pointer with '__counted_by_or_null' and the argument of the attribute must be defined in the same translation unit}}
extern int *__counted_by_or_null(len2) extptr4;
// expected-error@+1{{pointer with '__sized_by_or_null' and the argument of the attribute must be defined in the same translation unit}}
extern void *__sized_by_or_null(len2) extptr5;

extern int redecl_len1;
extern void *__sized_by(redecl_len1) redecl_ptr1;
int redecl_len1;
void *__sized_by(redecl_len1) redecl_ptr1;

extern int redecl_len2;
extern int *__sized_by(redecl_len2) redecl_ptr2; // expected-note{{'redecl_ptr2' declared here}}
int redecl_len2;
int *__counted_by(redecl_len2) redecl_ptr2;
// expected-error@-1{{conflicting '__counted_by' attribute with the previous variable declaration}}

extern int redecl_len3;
extern int *redecl_ptr3; // expected-note 4{{'redecl_ptr3' declared here}}
int redecl_len3;
int *__counted_by(redecl_len3) redecl_ptr3;
// expected-error@-1{{conflicting '__counted_by' attribute with the previous variable declaration}}
int *__sized_by(redecl_len3) redecl_ptr3;
// expected-error@-1{{conflicting '__sized_by' attribute with the previous variable declaration}}
int *__counted_by_or_null(redecl_len3) redecl_ptr3;
// expected-error@-1{{conflicting '__counted_by_or_null' attribute with the previous variable declaration}}
int *__sized_by_or_null(redecl_len3) redecl_ptr3;
// expected-error@-1{{conflicting '__sized_by_or_null' attribute with the previous variable declaration}}

extern int redecl_len4;
extern int *__counted_by(redecl_len4) redecl_ptr4;
extern int redecl_len5;
extern int *__counted_by(redecl_len5) redecl_ptr5; // expected-note 4{{'redecl_ptr5' declared here}}
int redecl_len4;
int *__counted_by(redecl_len4) redecl_ptr5;
// expected-error@-1{{conflicting '__counted_by' attribute with the previous variable declaration}}
int *__sized_by(redecl_len4) redecl_ptr5;
// expected-error@-1{{conflicting '__sized_by' attribute with the previous variable declaration}}
int *__counted_by_or_null(redecl_len4) redecl_ptr5;
// expected-error@-1{{conflicting '__counted_by_or_null' attribute with the previous variable declaration}}
int *__sized_by_or_null(redecl_len4) redecl_ptr5;
// expected-error@-1{{conflicting '__sized_by_or_null' attribute with the previous variable declaration}}

int len_2 = 2;
int *__counted_by(len_2 - 2) ptr_with_expr_count;
