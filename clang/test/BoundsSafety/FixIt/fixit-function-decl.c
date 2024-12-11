
// RUN: %clang_cc1 -fbounds-safety -fblocks -verify -verify-ignore-unexpected=note %s
// RUN: cp %s %t
// RUN: not %clang_cc1 -fbounds-safety -fixit -fix-what-you-can %t 2> /dev/null
// RUN: grep -v CHECK %t | FileCheck %s

#include <ptrcheck.h>

// CHECK: F0RT __bidi_indexable f0(void);
// CHECK: int *__bidi_indexable f0(void);
// expected-error@+3{{conflicting types for 'f0'}}
#define F0RT int *
F0RT f0(void);
int *__bidi_indexable f0(void);

// CHECK: F1RT f1(void);
// CHECK: int *__bidi_indexable *f1(void);
// expected-error@+3{{conflicting types for 'f1'}}
#define F1RT int **
F1RT f1(void); // no fixit: can't add bidi_indexable in the middle
int *__bidi_indexable *f1(void);

// CHECK: int f2(void);
// CHECK: int *__bidi_indexable *f2(void);
// expected-error@+2{{conflicting types for 'f2'}}
int f2(void); // no fixit: can't change from non-pointer to pointer type
int *__bidi_indexable *f2(void);

// CHECK: int *__bidi_indexable f3(void);
// CHECK: int *__bidi_indexable f3(void);
// expected-error@+2{{conflicting types for 'f3'}}
int *f3(void);
int *__bidi_indexable f3(void);

// CHECK: f4rt __bidi_indexable f4(void);
// CHECK: int *__bidi_indexable f4(void);
// expected-error@+3{{conflicting types for 'f4'}}
typedef int *f4rt;
f4rt f4(void);
int *__bidi_indexable f4(void);

// CHECK: f5rt f5(void);
// CHECK: int *__bidi_indexable *f5(void);
// expected-error@+3{{conflicting types for 'f5'}}
typedef int **f5rt;
f5rt f5(void); // no fixit: can't add bidi_indexable in the middle
int *__bidi_indexable *f5(void);

// CHECK: void f6(int *b, int a);
// CHECK: void f6(int a, int *__counted_by(a) b);
// expected-error@+2{{conflicting types for 'f6'}}
void f6(int *b, int a);
void f6(int a, int *__counted_by(a) b); // no fixit: swapped arguments

// CHECK: void f7(int *a);
// CHECK: void f7(int *__counted_by(b) a, int b);
// expected-error@+2{{conflicting '__counted_by' attribute with the previous function declaration}}
void f7(int *a);
void f7(int *__counted_by(b) a, int b); // no fixit: added argument

// CHECK: void f8(int *a, int b, int c);
// CHECK: void f8(int *__counted_by(b) a, int b);
// expected-error@+2{{conflicting '__counted_by' attribute with the previous function declaration}}
void f8(int *a, int b, int c);
void f8(int *__counted_by(b) a, int b); // no fixit: removed argument

// CHECK: void f9(int *__counted_by(b) a, int b);
// CHECK: void f9(int *__counted_by(b) a, int b);
// expected-error@+2{{conflicting '__counted_by' attribute with the previous function declaration}}
void f9(int *a, int b);
void f9(int *__counted_by(b) a, int b);

// CHECK: void f10(int *a, int b);
// CHECK: void f10(int *__counted_by(hello) a, int hello);
// expected-error@+2{{conflicting '__counted_by' attribute with the previous function declaration}}
void f10(int *a, int b);
void f10(int *__counted_by(hello) a, int hello); // TODO: no fixit because variable name changed :(

// CHECK: void f11(int *a, int b, int c);
// CHECK: void f11(int *__counted_by(hello + world) a, int hello, int world);
// expected-error@+2{{conflicting '__counted_by' attribute with the previous function declaration}}
void f11(int *a, int b, int c);
void f11(int *__counted_by(hello + world) a, int hello, int world); // TODO: no fixit because variable name changed :(

// CHECK: void f12(int *__counted_by(b + c) a, int b, int c);
// CHECK: void f12(int *__counted_by(b + c) a, int b, int c);
// expected-error@+2{{conflicting '__counted_by' attribute with the previous function declaration}}
void f12(int *a, int b, int c);
void f12(int *__counted_by(b + c) a, int b, int c);

// CHECK: int *__counted_by(count) f13(int count);
// CHECK: int *__counted_by(count) f13(int count);
// expected-error@+2{{conflicting '__counted_by' attribute with the previous function declaration}}
int *f13(int count);
int *__counted_by(count) f13(int count);

// CHECK: void f14(int *__sized_by(b) a, int b);
// CHECK: void f14(int *__sized_by(b) a, int b);
// expected-error@+2{{conflicting '__sized_by' attribute with the previous function declaration}}
void f14(int *a, int b);
void f14(int *__sized_by(b) a, int b);

// CHECK: void f15(int *__sized_by(b + c) a, int b, int c);
// CHECK: void f15(int *__sized_by(b + c) a, int b, int c);
// expected-error@+2{{conflicting '__sized_by' attribute with the previous function declaration}}
void f15(int *a, int b, int c);
void f15(int *__sized_by(b + c) a, int b, int c);

// CHECK: int *__sized_by(count) f16(int count);
// CHECK: int *__sized_by(count) f16(int count);
// expected-error@+2{{conflicting '__sized_by' attribute with the previous function declaration}}
int *f16(int count);
int *__sized_by(count) f16(int count);

// CHECK: void f17(int *__ended_by(b) a, int *b);
// CHECK: void f17(int *__ended_by(b) a, int *b);
// expected-error@+2{{conflicting '__ended_by' attribute with the previous function declaration}}
void f17(int *a, int *b);
void f17(int *__ended_by(b) a, int *b);

// CHECK: int *f18(int *a);
// CHECK: typeof(f18) f18;
int *f18(int *a);
typeof(f18) f18;

// CHECK: int *__counted_by(b) f19(int *__counted_by(b) a, int b);
// CHECK: typeof(f19) f20;
// CHECK: int *__counted_by(b) f20(int *__counted_by(b) a, int b);
// expected-error@+3{{conflicting '__counted_by' attribute with the previous function declaration}}
int *__counted_by(b) f19(int *__counted_by(b) a, int b);
typeof(f19) f20;
int *f20(int *a, int b);

// CHECK: void f21(int a, int *__counted_by(a) b);
// CHECK: void f21(int a, int *__counted_by(a) b);
// expected-error@+2{{conflicting '__counted_by' attribute with the previous function declaration}}
void f21(int a, int *b);
void f21(int a, int *__counted_by(a) b);

// CHECK: void f22(F0PT __bidi_indexable p);
// CHECK: void f22(int *__bidi_indexable p);
// expected-error@+3{{conflicting types for 'f22'}}
#define F0PT int *
void f22(F0PT p);
void f22(int *__bidi_indexable p);

// CHECK: void f23(F1PT p);
// CHECK: void f23(int *__bidi_indexable *p);
// expected-error@+3{{conflicting types for 'f23'}}
#define F1PT int **
void f23(F1PT p); // no fixit: can't add bidi_indexable in the middle
void f23(int *__bidi_indexable *p);

// CHECK: void f24(int *__indexable p);
// CHECK: void f24(int *__indexable p);
// expected-error@+2{{conflicting types for 'f24'}}
void f24(int *p);
void f24(int *__indexable p);

// CHECK: fptr_t f25;
// CHECK: void f25(int *__bidi_indexable p);
// expected-error@+3{{conflicting types for 'f25'}}
typedef void fptr_t(int *p);
fptr_t f25;
void f25(int *__bidi_indexable p);

// CHECK: void f26(int *__bidi_indexable p);
// CHECK: void f26(int *__bidi_indexable p) {
// expected-error@+2{{conflicting types for 'f26'}}
void f26(int *p);
void f26(int *__bidi_indexable p) {
    return;
}

// CHECK: void f27(int *__bidi_indexable p);
// CHECK: void f27(int *__bidi_indexable p);
// expected-error@+2{{conflicting types for 'f27'}}
void f27(int *__bidi_indexable p);
void f27(int *p);

// CHECK: int* __bidi_indexable f28(void);
// CHECK: int* __bidi_indexable f28(void);
// expected-error@+2{{conflicting types for 'f28'}}
int* __bidi_indexable f28(void);
int* f28(void);

// CHECK: int* __counted_by(count) f29(int count);
// CHECK: int* __counted_by(count) f29(int count);
// expected-error@+2{{conflicting '__counted_by' attribute with the previous function declaration}}
int* __counted_by(count) f29(int count);
int* f29(int count);

// CHECK: int f30(int* __bidi_indexable b);
// CHECK: int f30(int* __bidi_indexable b);
// expected-error@+2{{conflicting types for 'f30'}}
int f30(int* __bidi_indexable b);
int f30(int* b);

// CHECK: int f31(int* __bidi_indexable);
// CHECK: int f31(int*__bidi_indexable);
// expected-error@+2{{conflicting types for 'f31'}}
int f31(int* __bidi_indexable);
int f31(int*);

// CHECK: int f32(int *__bidi_indexable buffer);
// CHECK: int f32(int *__bidi_indexable buffer);
// expected-error@+2{{conflicting types for 'f32'}}
int f32(int *__bidi_indexable buffer);
int f32(int *buffer);

// CHECK: int f33(int* __bidi_indexable buffer);
// CHECK: int f33(int* __bidi_indexable buffer);
// expected-error@+2{{conflicting types for 'f33'}}
int f33(int* __bidi_indexable buffer);
int f33(int* buffer);

// CHECK: int f34(int *__counted_by(size) buffer, int size);
// CHECK: int f34(int *__counted_by(size) buffer, int size);
// expected-error@+2{{conflicting '__counted_by' attribute with the previous function declaration}}
int f34(int *__counted_by(size) buffer, int size);
int f34(int *buffer, int size);

// CHECK: int f35(int *nullable __bidi_indexable foo);
// CHECK: int f35(int *nullable __bidi_indexable foo);
// expected-error@+3{{conflicting types for 'f35'}}
#define nullable _Nullable
int f35(int *nullable foo);
int f35(int *nullable __bidi_indexable foo);

// CHECK: int f36(int *nullable __bidi_indexable foo);
// CHECK: int f36(int *__bidi_indexable nullable foo);
// expected-error@+3{{conflicting types for 'f36'}}
#define nullable _Nullable
int f36(int *nullable foo);
int f36(int *__bidi_indexable nullable foo);

// CHECK: int f37(int *__bidi_indexable const foo);
// CHECK: int f37(int *const __bidi_indexable foo);
// expected-error@+2{{conflicting types for 'f37'}}
int f37(int *const foo);
int f37(int *const __bidi_indexable foo);

// CHECK: int f38(int *__bidi_indexable const foo);
// CHECK: int f38(int *__bidi_indexable const foo);
// expected-error@+2{{conflicting types for 'f38'}}
int f38(int *const foo);
int f38(int *__bidi_indexable const foo);

// CHECK: int f39(int *__bidi_indexable foo);
// CHECK: int f39(int *my_bidi foo);
// expected-error@+3{{conflicting types for 'f39'}}
#define my_bidi __bidi_indexable
int f39(int *foo);
int f39(int *my_bidi foo);

// CHECK: int *__bidi_indexable f40(int *__indexable p);
// CHECK: int *__bidi_indexable f40(int *__indexable p);
// expected-error@+2{{conflicting types for 'f40'}}
int *f40(int *p);
int *__bidi_indexable f40(int *__indexable p);

//==============================================================================
// _Nonnull attribute on pointers
//
// The expected application of fix-its below
//
// * Doesn't try to match mismatched `_Nonnull` attributes
// * Demonstrates several cases where fix-its fail to apply but should (rdar://116016096)
// * Demonstrates several cases where a fix-it is applied but creates broken code (rdar://116016096)
//
// The declarations below were programmatically generated from a 
// configuration matrix of:
//
// * Attribute is on parameter or return type
// * Parameter name attribute is on is specified
// * Attribute appears only on first or second decl
// * Attribute name (__counted_by and __bidi_indexable were used below)
// * Attribute is on inner or outer pointer
// * _Nonnull is on the destination type for the fix
// * _Nonnull is on the source type which the fix is derived from
//==============================================================================


// expected-error@+4{{conflicting '__counted_by' attribute with the previous function declaration}}
// CHECK: void f_on_param_named_param_attr_fd_cb_outer_ptr_nn_dest_nn_src(int *_Nonnull*_Nonnull __counted_by(len) buffer, unsigned len);
// CHECK: void f_on_param_named_param_attr_fd_cb_outer_ptr_nn_dest_nn_src(int *_Nonnull*_Nonnull __counted_by(len) buffer, unsigned len);
void f_on_param_named_param_attr_fd_cb_outer_ptr_nn_dest_nn_src(int *_Nonnull*_Nonnull __counted_by(len) buffer, unsigned len);
void f_on_param_named_param_attr_fd_cb_outer_ptr_nn_dest_nn_src(int *_Nonnull*_Nonnull buffer, unsigned len);

// This catches a crash (rdar://115575540) where the `_Nonnull`
// attribute wasn't handled.
// expected-error@+4{{conflicting '__counted_by' attribute with the previous function declaration}}
// CHECK: void f_on_param_named_param_attr_fd_cb_outer_ptr_nn_dest(int ** __counted_by(len) buffer, unsigned len);
// CHECK: void f_on_param_named_param_attr_fd_cb_outer_ptr_nn_dest(int *_Nonnull*_Nonnull __counted_by(len) buffer, unsigned len);
void f_on_param_named_param_attr_fd_cb_outer_ptr_nn_dest(int ** __counted_by(len) buffer, unsigned len);
void f_on_param_named_param_attr_fd_cb_outer_ptr_nn_dest(int *_Nonnull*_Nonnull buffer, unsigned len);

// expected-error@+4{{conflicting '__counted_by' attribute with the previous function declaration}}
// CHECK: void f_on_param_named_param_attr_fd_cb_outer_ptr_nn_src(int *_Nonnull*_Nonnull __counted_by(len) buffer, unsigned len);
// CHECK: void f_on_param_named_param_attr_fd_cb_outer_ptr_nn_src(int ** __counted_by(len) buffer, unsigned len);
void f_on_param_named_param_attr_fd_cb_outer_ptr_nn_src(int *_Nonnull*_Nonnull __counted_by(len) buffer, unsigned len);
void f_on_param_named_param_attr_fd_cb_outer_ptr_nn_src(int ** buffer, unsigned len);

// TODO(dliew): A fix-it should be applied here (rdar://116016096).
// expected-error@+4{{conflicting '__counted_by' attribute with the previous function declaration}}
// CHECK: void f_on_param_named_param_attr_fd_cb_inner_ptr_nn_dest_nn_src(int *_Nonnull __counted_by(len)*_Nonnull buffer, unsigned len);
// CHECK: void f_on_param_named_param_attr_fd_cb_inner_ptr_nn_dest_nn_src(int *_Nonnull*_Nonnull buffer, unsigned len);
void f_on_param_named_param_attr_fd_cb_inner_ptr_nn_dest_nn_src(int *_Nonnull __counted_by(len)*_Nonnull buffer, unsigned len);
void f_on_param_named_param_attr_fd_cb_inner_ptr_nn_dest_nn_src(int *_Nonnull*_Nonnull buffer, unsigned len);

// TODO(dliew): A fix-it should be applied here (rdar://116016096).
// expected-error@+4{{conflicting '__counted_by' attribute with the previous function declaration}}
// CHECK: void f_on_param_named_param_attr_fd_cb_inner_ptr_nn_dest(int * __counted_by(len)* buffer, unsigned len);
// CHECK: void f_on_param_named_param_attr_fd_cb_inner_ptr_nn_dest(int *_Nonnull*_Nonnull buffer, unsigned len);
void f_on_param_named_param_attr_fd_cb_inner_ptr_nn_dest(int * __counted_by(len)* buffer, unsigned len);
void f_on_param_named_param_attr_fd_cb_inner_ptr_nn_dest(int *_Nonnull*_Nonnull buffer, unsigned len);

// TODO(dliew): A fix-it should be applied here (rdar://116016096).
// expected-error@+4{{conflicting '__counted_by' attribute with the previous function declaration}}
// CHECK: void f_on_param_named_param_attr_fd_cb_inner_ptr_nn_src(int *_Nonnull __counted_by(len)*_Nonnull buffer, unsigned len);
// CHECK: void f_on_param_named_param_attr_fd_cb_inner_ptr_nn_src(int ** buffer, unsigned len); 
void f_on_param_named_param_attr_fd_cb_inner_ptr_nn_src(int *_Nonnull __counted_by(len)*_Nonnull buffer, unsigned len);
void f_on_param_named_param_attr_fd_cb_inner_ptr_nn_src(int ** buffer, unsigned len);

// expected-error@+4{{conflicting types for 'f_on_param_named_param_attr_fd_b_outer_ptr_nn_dest_nn_src'}}
// CHECK: void f_on_param_named_param_attr_fd_b_outer_ptr_nn_dest_nn_src(int *_Nonnull*_Nonnull __bidi_indexable buffer);
// CHECK: void f_on_param_named_param_attr_fd_b_outer_ptr_nn_dest_nn_src(int *_Nonnull*_Nonnull __bidi_indexable buffer);
void f_on_param_named_param_attr_fd_b_outer_ptr_nn_dest_nn_src(int *_Nonnull*_Nonnull __bidi_indexable buffer);
void f_on_param_named_param_attr_fd_b_outer_ptr_nn_dest_nn_src(int *_Nonnull*_Nonnull buffer);

// expected-error@+4{{conflicting types for 'f_on_param_named_param_attr_fd_b_outer_ptr_nn_dest'}}
// CHECK: void f_on_param_named_param_attr_fd_b_outer_ptr_nn_dest(int ** __bidi_indexable buffer);
// CHECK: void f_on_param_named_param_attr_fd_b_outer_ptr_nn_dest(int *_Nonnull*_Nonnull __bidi_indexable buffer);
void f_on_param_named_param_attr_fd_b_outer_ptr_nn_dest(int ** __bidi_indexable buffer);
void f_on_param_named_param_attr_fd_b_outer_ptr_nn_dest(int *_Nonnull*_Nonnull buffer);

// expected-error@+4{{conflicting types for 'f_on_param_named_param_attr_fd_b_outer_ptr_nn_src'}}
// CHECK: void f_on_param_named_param_attr_fd_b_outer_ptr_nn_src(int *_Nonnull*_Nonnull __bidi_indexable buffer);
// CHECK: void f_on_param_named_param_attr_fd_b_outer_ptr_nn_src(int ** __bidi_indexable buffer);
void f_on_param_named_param_attr_fd_b_outer_ptr_nn_src(int *_Nonnull*_Nonnull __bidi_indexable buffer);
void f_on_param_named_param_attr_fd_b_outer_ptr_nn_src(int ** buffer);

// TODO(dliew): A fix-it should be applied here (rdar://116016096).
// expected-error@+4{{conflicting types for 'f_on_param_named_param_attr_fd_b_inner_ptr_nn_dest_nn_src'}}
// CHECK: void f_on_param_named_param_attr_fd_b_inner_ptr_nn_dest_nn_src(int *_Nonnull __bidi_indexable*_Nonnull buffer);
// CHECK: void f_on_param_named_param_attr_fd_b_inner_ptr_nn_dest_nn_src(int *_Nonnull*_Nonnull buffer);
void f_on_param_named_param_attr_fd_b_inner_ptr_nn_dest_nn_src(int *_Nonnull __bidi_indexable*_Nonnull buffer);
void f_on_param_named_param_attr_fd_b_inner_ptr_nn_dest_nn_src(int *_Nonnull*_Nonnull buffer);

// TODO(dliew): A fix-it should be applied here (rdar://116016096).
// expected-error@+4{{conflicting types for 'f_on_param_named_param_attr_fd_b_inner_ptr_nn_dest'}}
// CHECK: void f_on_param_named_param_attr_fd_b_inner_ptr_nn_dest(int * __bidi_indexable* buffer);
// CHECK: void f_on_param_named_param_attr_fd_b_inner_ptr_nn_dest(int *_Nonnull*_Nonnull buffer);
void f_on_param_named_param_attr_fd_b_inner_ptr_nn_dest(int * __bidi_indexable* buffer);
void f_on_param_named_param_attr_fd_b_inner_ptr_nn_dest(int *_Nonnull*_Nonnull buffer);

// TODO(dliew): A fix-it should be applied here (rdar://116016096).
// expected-error@+4{{conflicting types for 'f_on_param_named_param_attr_fd_b_inner_ptr_nn_src'}}
// CHECK: void f_on_param_named_param_attr_fd_b_inner_ptr_nn_src(int *_Nonnull __bidi_indexable*_Nonnull buffer);
// CHECK: void f_on_param_named_param_attr_fd_b_inner_ptr_nn_src(int ** buffer);
void f_on_param_named_param_attr_fd_b_inner_ptr_nn_src(int *_Nonnull __bidi_indexable*_Nonnull buffer);
void f_on_param_named_param_attr_fd_b_inner_ptr_nn_src(int ** buffer);

// expected-error@+4{{conflicting '__counted_by' attribute with the previous function declaration}}
// CHECK: void f_on_param_named_param_attr_sd_cb_outer_ptr_nn_dest_nn_src(int *_Nonnull*_Nonnull __counted_by(len) buffer, unsigned len);
// CHECK: void f_on_param_named_param_attr_sd_cb_outer_ptr_nn_dest_nn_src(int *_Nonnull*_Nonnull __counted_by(len) buffer, unsigned len);
void f_on_param_named_param_attr_sd_cb_outer_ptr_nn_dest_nn_src(int *_Nonnull*_Nonnull buffer, unsigned len);
void f_on_param_named_param_attr_sd_cb_outer_ptr_nn_dest_nn_src(int *_Nonnull*_Nonnull __counted_by(len) buffer, unsigned len);

// expected-error@+4{{conflicting '__counted_by' attribute with the previous function declaration}}
// CHECK: void f_on_param_named_param_attr_sd_cb_outer_ptr_nn_dest(int *_Nonnull*_Nonnull __counted_by(len) buffer, unsigned len);
// CHECK: void f_on_param_named_param_attr_sd_cb_outer_ptr_nn_dest(int ** __counted_by(len) buffer, unsigned len);
void f_on_param_named_param_attr_sd_cb_outer_ptr_nn_dest(int *_Nonnull*_Nonnull buffer, unsigned len);
void f_on_param_named_param_attr_sd_cb_outer_ptr_nn_dest(int ** __counted_by(len) buffer, unsigned len);

// expected-error@+4{{conflicting '__counted_by' attribute with the previous function declaration}}
// CHECK: void f_on_param_named_param_attr_sd_cb_outer_ptr_nn_src(int ** __counted_by(len) buffer, unsigned len);
// CHECK: void f_on_param_named_param_attr_sd_cb_outer_ptr_nn_src(int *_Nonnull*_Nonnull __counted_by(len) buffer, unsigned len);
void f_on_param_named_param_attr_sd_cb_outer_ptr_nn_src(int ** buffer, unsigned len);
void f_on_param_named_param_attr_sd_cb_outer_ptr_nn_src(int *_Nonnull*_Nonnull __counted_by(len) buffer, unsigned len);

// TODO(dliew): A fix-it should be applied here (rdar://116016096).
// expected-error@+4{{conflicting '__counted_by' attribute with the previous function declaration}}
// CHECK: void f_on_param_named_param_attr_sd_cb_inner_ptr_nn_dest_nn_src(int *_Nonnull*_Nonnull buffer, unsigned len);
// CHECK: void f_on_param_named_param_attr_sd_cb_inner_ptr_nn_dest_nn_src(int *_Nonnull __counted_by(len)*_Nonnull buffer, unsigned len);
void f_on_param_named_param_attr_sd_cb_inner_ptr_nn_dest_nn_src(int *_Nonnull*_Nonnull buffer, unsigned len);
void f_on_param_named_param_attr_sd_cb_inner_ptr_nn_dest_nn_src(int *_Nonnull __counted_by(len)*_Nonnull buffer, unsigned len);

// TODO(dliew): A fix-it should be applied here (rdar://116016096).
// expected-error@+4{{conflicting '__counted_by' attribute with the previous function declaration}}
// CHECK: void f_on_param_named_param_attr_sd_cb_inner_ptr_nn_dest(int *_Nonnull*_Nonnull buffer, unsigned len);
// CHECK: void f_on_param_named_param_attr_sd_cb_inner_ptr_nn_dest(int * __counted_by(len)* buffer, unsigned len);
void f_on_param_named_param_attr_sd_cb_inner_ptr_nn_dest(int *_Nonnull*_Nonnull buffer, unsigned len);
void f_on_param_named_param_attr_sd_cb_inner_ptr_nn_dest(int * __counted_by(len)* buffer, unsigned len);

// TODO(dliew): A fix-it should be applied here (rdar://116016096).
// expected-error@+4{{conflicting '__counted_by' attribute with the previous function declaration}}
// CHECK: void f_on_param_named_param_attr_sd_cb_inner_ptr_nn_src(int ** buffer, unsigned len);
// CHECK: void f_on_param_named_param_attr_sd_cb_inner_ptr_nn_src(int *_Nonnull __counted_by(len)*_Nonnull buffer, unsigned len);
void f_on_param_named_param_attr_sd_cb_inner_ptr_nn_src(int ** buffer, unsigned len);
void f_on_param_named_param_attr_sd_cb_inner_ptr_nn_src(int *_Nonnull __counted_by(len)*_Nonnull buffer, unsigned len);

// expected-error@+4{{conflicting types for 'f_on_param_named_param_attr_sd_b_outer_ptr_nn_dest_nn_src'}}
// CHECK: void f_on_param_named_param_attr_sd_b_outer_ptr_nn_dest_nn_src(int *_Nonnull*_Nonnull __bidi_indexable buffer);
// CHECK: void f_on_param_named_param_attr_sd_b_outer_ptr_nn_dest_nn_src(int *_Nonnull*_Nonnull __bidi_indexable buffer);
void f_on_param_named_param_attr_sd_b_outer_ptr_nn_dest_nn_src(int *_Nonnull*_Nonnull buffer);
void f_on_param_named_param_attr_sd_b_outer_ptr_nn_dest_nn_src(int *_Nonnull*_Nonnull __bidi_indexable buffer);

// expected-error@+4{{conflicting types for 'f_on_param_named_param_attr_sd_b_outer_ptr_nn_dest'}}
// CHECK: void f_on_param_named_param_attr_sd_b_outer_ptr_nn_dest(int *_Nonnull*_Nonnull __bidi_indexable buffer);
// CHECK: void f_on_param_named_param_attr_sd_b_outer_ptr_nn_dest(int ** __bidi_indexable buffer);
void f_on_param_named_param_attr_sd_b_outer_ptr_nn_dest(int *_Nonnull*_Nonnull buffer);
void f_on_param_named_param_attr_sd_b_outer_ptr_nn_dest(int ** __bidi_indexable buffer);

// expected-error@+4{{conflicting types for 'f_on_param_named_param_attr_sd_b_outer_ptr_nn_src'}}
// CHECK: void f_on_param_named_param_attr_sd_b_outer_ptr_nn_src(int ** __bidi_indexable buffer);
// CHECK: void f_on_param_named_param_attr_sd_b_outer_ptr_nn_src(int *_Nonnull*_Nonnull __bidi_indexable buffer);
void f_on_param_named_param_attr_sd_b_outer_ptr_nn_src(int ** buffer);
void f_on_param_named_param_attr_sd_b_outer_ptr_nn_src(int *_Nonnull*_Nonnull __bidi_indexable buffer);

// TODO(dliew): A fix-it should be applied here (rdar://116016096).
// expected-error@+4{{conflicting types for 'f_on_param_named_param_attr_sd_b_inner_ptr_nn_dest_nn_src'}}
// CHECK: void f_on_param_named_param_attr_sd_b_inner_ptr_nn_dest_nn_src(int *_Nonnull*_Nonnull buffer);
// CHECK: void f_on_param_named_param_attr_sd_b_inner_ptr_nn_dest_nn_src(int *_Nonnull __bidi_indexable*_Nonnull buffer);
void f_on_param_named_param_attr_sd_b_inner_ptr_nn_dest_nn_src(int *_Nonnull*_Nonnull buffer);
void f_on_param_named_param_attr_sd_b_inner_ptr_nn_dest_nn_src(int *_Nonnull __bidi_indexable*_Nonnull buffer);

// TODO(dliew): A fix-it should be applied here (rdar://116016096).
// expected-error@+4{{conflicting types for 'f_on_param_named_param_attr_sd_b_inner_ptr_nn_dest'}}
// CHECK: void f_on_param_named_param_attr_sd_b_inner_ptr_nn_dest(int *_Nonnull*_Nonnull buffer);
// CHECK: void f_on_param_named_param_attr_sd_b_inner_ptr_nn_dest(int * __bidi_indexable* buffer);
void f_on_param_named_param_attr_sd_b_inner_ptr_nn_dest(int *_Nonnull*_Nonnull buffer);
void f_on_param_named_param_attr_sd_b_inner_ptr_nn_dest(int * __bidi_indexable* buffer);

// TODO(dliew): A fix-it should be applied here (rdar://116016096).
// expected-error@+4{{conflicting types for 'f_on_param_named_param_attr_sd_b_inner_ptr_nn_src'}}
// CHECK: void f_on_param_named_param_attr_sd_b_inner_ptr_nn_src(int ** buffer);
// CHECK: void f_on_param_named_param_attr_sd_b_inner_ptr_nn_src(int *_Nonnull __bidi_indexable*_Nonnull buffer);
void f_on_param_named_param_attr_sd_b_inner_ptr_nn_src(int ** buffer);
void f_on_param_named_param_attr_sd_b_inner_ptr_nn_src(int *_Nonnull __bidi_indexable*_Nonnull buffer);

// TODO(dliew): A fix-it applied here but it results in broken code (rdar://116016096).
// expected-error@+4{{conflicting '__counted_by' attribute with the previous function declaration}}
// CHECK: void f_on_param_attr_fd_cb_outer_ptr_nn_dest_nn_src(int *_Nonnull*_Nonnull __counted_by(len), unsigned len);
// CHECK: void f_on_param_attr_fd_cb_outer_ptr_nn_dest_nn_src(int *_Nonnull*_Nonnull__counted_by(len) , unsigned len);
void f_on_param_attr_fd_cb_outer_ptr_nn_dest_nn_src(int *_Nonnull*_Nonnull __counted_by(len), unsigned len);
void f_on_param_attr_fd_cb_outer_ptr_nn_dest_nn_src(int *_Nonnull*_Nonnull, unsigned len);

// TODO(dliew): A fix-it applied here but it results in broken code (rdar://116016096).
// expected-error@+4{{conflicting '__counted_by' attribute with the previous function declaration}}
// CHECK: void f_on_param_attr_fd_cb_outer_ptr_nn_dest(int ** __counted_by(len), unsigned len);
// CHECK: void f_on_param_attr_fd_cb_outer_ptr_nn_dest(int *_Nonnull*_Nonnull__counted_by(len) , unsigned len);
void f_on_param_attr_fd_cb_outer_ptr_nn_dest(int ** __counted_by(len), unsigned len);
void f_on_param_attr_fd_cb_outer_ptr_nn_dest(int *_Nonnull*_Nonnull, unsigned len);

// expected-error@+4{{conflicting '__counted_by' attribute with the previous function declaration}}
// CHECK: void f_on_param_attr_fd_cb_outer_ptr_nn_src(int *_Nonnull*_Nonnull __counted_by(len), unsigned len);
// CHECK: void f_on_param_attr_fd_cb_outer_ptr_nn_src(int **__counted_by(len), unsigned len);
void f_on_param_attr_fd_cb_outer_ptr_nn_src(int *_Nonnull*_Nonnull __counted_by(len), unsigned len);
void f_on_param_attr_fd_cb_outer_ptr_nn_src(int **, unsigned len);

// TODO(dliew): A fix-it should be applied here (rdar://116016096).
// expected-error@+4{{conflicting '__counted_by' attribute with the previous function declaration}}
// CHECK: void f_on_param_attr_fd_cb_inner_ptr_nn_dest_nn_src(int *_Nonnull __counted_by(len)*_Nonnull, unsigned len);
// CHECK: void f_on_param_attr_fd_cb_inner_ptr_nn_dest_nn_src(int *_Nonnull*_Nonnull, unsigned len);
void f_on_param_attr_fd_cb_inner_ptr_nn_dest_nn_src(int *_Nonnull __counted_by(len)*_Nonnull, unsigned len);
void f_on_param_attr_fd_cb_inner_ptr_nn_dest_nn_src(int *_Nonnull*_Nonnull, unsigned len);

// TODO(dliew): A fix-it should be applied here (rdar://116016096).
// expected-error@+4{{conflicting '__counted_by' attribute with the previous function declaration}}
// CHECK: void f_on_param_attr_fd_cb_inner_ptr_nn_dest(int * __counted_by(len)*, unsigned len);
// CHECK: void f_on_param_attr_fd_cb_inner_ptr_nn_dest(int *_Nonnull*_Nonnull, unsigned len);
void f_on_param_attr_fd_cb_inner_ptr_nn_dest(int * __counted_by(len)*, unsigned len);
void f_on_param_attr_fd_cb_inner_ptr_nn_dest(int *_Nonnull*_Nonnull, unsigned len);

// TODO(dliew): A fix-it should be applied here (rdar://116016096).
// expected-error@+4{{conflicting '__counted_by' attribute with the previous function declaration}}
// CHECK: void f_on_param_attr_fd_cb_inner_ptr_nn_src(int *_Nonnull __counted_by(len)*_Nonnull, unsigned len);
// CHECK: void f_on_param_attr_fd_cb_inner_ptr_nn_src(int **, unsigned len);
void f_on_param_attr_fd_cb_inner_ptr_nn_src(int *_Nonnull __counted_by(len)*_Nonnull, unsigned len);
void f_on_param_attr_fd_cb_inner_ptr_nn_src(int **, unsigned len);

// TODO(dliew): A fix-it applied here but it results in broken code (rdar://116016096).
// expected-error@+4{{conflicting types for 'f_on_param_attr_fd_b_outer_ptr_nn_dest_nn_src'}}
// CHECK: void f_on_param_attr_fd_b_outer_ptr_nn_dest_nn_src(int *_Nonnull*_Nonnull __bidi_indexable);
// CHECK: void f_on_param_attr_fd_b_outer_ptr_nn_dest_nn_src(int *_Nonnull*_Nonnull__bidi_indexable );
void f_on_param_attr_fd_b_outer_ptr_nn_dest_nn_src(int *_Nonnull*_Nonnull __bidi_indexable);
void f_on_param_attr_fd_b_outer_ptr_nn_dest_nn_src(int *_Nonnull*_Nonnull);

// TODO(dliew): A fix-it applied here but it results in broken code (rdar://116016096).
// expected-error@+4{{conflicting types for 'f_on_param_attr_fd_b_outer_ptr_nn_dest'}}
// CHECK: void f_on_param_attr_fd_b_outer_ptr_nn_dest(int ** __bidi_indexable);
// CHECK: void f_on_param_attr_fd_b_outer_ptr_nn_dest(int *_Nonnull*_Nonnull__bidi_indexable );
void f_on_param_attr_fd_b_outer_ptr_nn_dest(int ** __bidi_indexable);
void f_on_param_attr_fd_b_outer_ptr_nn_dest(int *_Nonnull*_Nonnull);

// expected-error@+4{{conflicting types for 'f_on_param_attr_fd_b_outer_ptr_nn_src'}}
// CHECK: void f_on_param_attr_fd_b_outer_ptr_nn_src(int *_Nonnull*_Nonnull __bidi_indexable);
// CHECK: void f_on_param_attr_fd_b_outer_ptr_nn_src(int **__bidi_indexable);
void f_on_param_attr_fd_b_outer_ptr_nn_src(int *_Nonnull*_Nonnull __bidi_indexable);
void f_on_param_attr_fd_b_outer_ptr_nn_src(int **);

// TODO(dliew): A fix-it should be applied here (rdar://116016096).
// expected-error@+4{{conflicting types for 'f_on_param_attr_fd_b_inner_ptr_nn_dest_nn_src'}}
// CHECK: void f_on_param_attr_fd_b_inner_ptr_nn_dest_nn_src(int *_Nonnull __bidi_indexable*_Nonnull);
// CHECK: void f_on_param_attr_fd_b_inner_ptr_nn_dest_nn_src(int *_Nonnull*_Nonnull);
void f_on_param_attr_fd_b_inner_ptr_nn_dest_nn_src(int *_Nonnull __bidi_indexable*_Nonnull);
void f_on_param_attr_fd_b_inner_ptr_nn_dest_nn_src(int *_Nonnull*_Nonnull);

// TODO(dliew): A fix-it should be applied here (rdar://116016096).
// expected-error@+4{{conflicting types for 'f_on_param_attr_fd_b_inner_ptr_nn_dest'}}
// CHECK: void f_on_param_attr_fd_b_inner_ptr_nn_dest(int * __bidi_indexable*);
// CHECK: void f_on_param_attr_fd_b_inner_ptr_nn_dest(int *_Nonnull*_Nonnull);
void f_on_param_attr_fd_b_inner_ptr_nn_dest(int * __bidi_indexable*);
void f_on_param_attr_fd_b_inner_ptr_nn_dest(int *_Nonnull*_Nonnull);

// TODO(dliew): A fix-it should be applied here (rdar://116016096).
// expected-error@+4{{conflicting types for 'f_on_param_attr_fd_b_inner_ptr_nn_src'}}
// CHECK: void f_on_param_attr_fd_b_inner_ptr_nn_src(int *_Nonnull __bidi_indexable*_Nonnull);
// CHECK: void f_on_param_attr_fd_b_inner_ptr_nn_src(int **);
void f_on_param_attr_fd_b_inner_ptr_nn_src(int *_Nonnull __bidi_indexable*_Nonnull);
void f_on_param_attr_fd_b_inner_ptr_nn_src(int **);

// TODO(dliew): A fix-it applied here but it results in broken code (rdar://116016096).
// expected-error@+4{{conflicting '__counted_by' attribute with the previous function declaration}}
// CHECK: void f_on_param_attr_sd_cb_outer_ptr_nn_dest_nn_src(int *_Nonnull*_Nonnull__counted_by(len) , unsigned len);
// CHECK: void f_on_param_attr_sd_cb_outer_ptr_nn_dest_nn_src(int *_Nonnull*_Nonnull __counted_by(len), unsigned len);
void f_on_param_attr_sd_cb_outer_ptr_nn_dest_nn_src(int *_Nonnull*_Nonnull, unsigned len);
void f_on_param_attr_sd_cb_outer_ptr_nn_dest_nn_src(int *_Nonnull*_Nonnull __counted_by(len), unsigned len);

// TODO(dliew): A fix-it applied here but it results in broken code (rdar://116016096).
// expected-error@+4{{conflicting '__counted_by' attribute with the previous function declaration}}
// CHECK: void f_on_param_attr_sd_cb_outer_ptr_nn_dest(int *_Nonnull*_Nonnull__counted_by(len) , unsigned len);
// CHECK: void f_on_param_attr_sd_cb_outer_ptr_nn_dest(int ** __counted_by(len), unsigned len);
void f_on_param_attr_sd_cb_outer_ptr_nn_dest(int *_Nonnull*_Nonnull, unsigned len);
void f_on_param_attr_sd_cb_outer_ptr_nn_dest(int ** __counted_by(len), unsigned len);

// expected-error@+4{{conflicting '__counted_by' attribute with the previous function declaration}}
// CHECK: void f_on_param_attr_sd_cb_outer_ptr_nn_src(int **__counted_by(len), unsigned len);
// CHECK: void f_on_param_attr_sd_cb_outer_ptr_nn_src(int *_Nonnull*_Nonnull __counted_by(len), unsigned len);
void f_on_param_attr_sd_cb_outer_ptr_nn_src(int **, unsigned len);
void f_on_param_attr_sd_cb_outer_ptr_nn_src(int *_Nonnull*_Nonnull __counted_by(len), unsigned len);

// TODO(dliew): A fix-it should be applied here (rdar://116016096).
// expected-error@+4{{conflicting '__counted_by' attribute with the previous function declaration}}
// CHECK: void f_on_param_attr_sd_cb_inner_ptr_nn_dest_nn_src(int *_Nonnull*_Nonnull, unsigned len);
// CHECK: void f_on_param_attr_sd_cb_inner_ptr_nn_dest_nn_src(int *_Nonnull __counted_by(len)*_Nonnull, unsigned len);
void f_on_param_attr_sd_cb_inner_ptr_nn_dest_nn_src(int *_Nonnull*_Nonnull, unsigned len);
void f_on_param_attr_sd_cb_inner_ptr_nn_dest_nn_src(int *_Nonnull __counted_by(len)*_Nonnull, unsigned len);

// TODO(dliew): A fix-it should be applied here (rdar://116016096).
// expected-error@+4{{conflicting '__counted_by' attribute with the previous function declaration}}
// CHECK: void f_on_param_attr_sd_cb_inner_ptr_nn_dest(int *_Nonnull*_Nonnull, unsigned len);
// CHECK: void f_on_param_attr_sd_cb_inner_ptr_nn_dest(int * __counted_by(len)*, unsigned len);
void f_on_param_attr_sd_cb_inner_ptr_nn_dest(int *_Nonnull*_Nonnull, unsigned len);
void f_on_param_attr_sd_cb_inner_ptr_nn_dest(int * __counted_by(len)*, unsigned len);

// TODO(dliew): A fix-it should be applied here (rdar://116016096).
// expected-error@+4{{conflicting '__counted_by' attribute with the previous function declaration}}
// CHECK: void f_on_param_attr_sd_cb_inner_ptr_nn_src(int **, unsigned len);
// CHECK: void f_on_param_attr_sd_cb_inner_ptr_nn_src(int *_Nonnull __counted_by(len)*_Nonnull, unsigned len);
void f_on_param_attr_sd_cb_inner_ptr_nn_src(int **, unsigned len);
void f_on_param_attr_sd_cb_inner_ptr_nn_src(int *_Nonnull __counted_by(len)*_Nonnull, unsigned len);

// TODO(dliew): A fix-it applied here but it results in broken code (rdar://116016096).
// expected-error@+4{{conflicting types for 'f_on_param_attr_sd_b_outer_ptr_nn_dest_nn_src'}}
// CHECK: void f_on_param_attr_sd_b_outer_ptr_nn_dest_nn_src(int *_Nonnull*_Nonnull__bidi_indexable );
// CHECK: void f_on_param_attr_sd_b_outer_ptr_nn_dest_nn_src(int *_Nonnull*_Nonnull __bidi_indexable);
void f_on_param_attr_sd_b_outer_ptr_nn_dest_nn_src(int *_Nonnull*_Nonnull);
void f_on_param_attr_sd_b_outer_ptr_nn_dest_nn_src(int *_Nonnull*_Nonnull __bidi_indexable);

// TODO(dliew): A fix-it applied here but it results in broken code (rdar://116016096).
// expected-error@+4{{conflicting types for 'f_on_param_attr_sd_b_outer_ptr_nn_dest'}}
// CHECK: void f_on_param_attr_sd_b_outer_ptr_nn_dest(int *_Nonnull*_Nonnull__bidi_indexable );
// CHECK: void f_on_param_attr_sd_b_outer_ptr_nn_dest(int ** __bidi_indexable);
void f_on_param_attr_sd_b_outer_ptr_nn_dest(int *_Nonnull*_Nonnull);
void f_on_param_attr_sd_b_outer_ptr_nn_dest(int ** __bidi_indexable);

// expected-error@+4{{conflicting types for 'f_on_param_attr_sd_b_outer_ptr_nn_src'}}
// CHECK: void f_on_param_attr_sd_b_outer_ptr_nn_src(int **__bidi_indexable);
// CHECK: void f_on_param_attr_sd_b_outer_ptr_nn_src(int *_Nonnull*_Nonnull __bidi_indexable);
void f_on_param_attr_sd_b_outer_ptr_nn_src(int **);
void f_on_param_attr_sd_b_outer_ptr_nn_src(int *_Nonnull*_Nonnull __bidi_indexable);

// TODO(dliew): A fix-it should be applied here (rdar://116016096).
// expected-error@+4{{conflicting types for 'f_on_param_attr_sd_b_inner_ptr_nn_dest_nn_src'}}
// CHECK: void f_on_param_attr_sd_b_inner_ptr_nn_dest_nn_src(int *_Nonnull*_Nonnull);
// CHECK: void f_on_param_attr_sd_b_inner_ptr_nn_dest_nn_src(int *_Nonnull __bidi_indexable*_Nonnull);
void f_on_param_attr_sd_b_inner_ptr_nn_dest_nn_src(int *_Nonnull*_Nonnull);
void f_on_param_attr_sd_b_inner_ptr_nn_dest_nn_src(int *_Nonnull __bidi_indexable*_Nonnull);

// TODO(dliew): A fix-it should be applied here (rdar://116016096).
// expected-error@+4{{conflicting types for 'f_on_param_attr_sd_b_inner_ptr_nn_dest'}}
// CHECK: void f_on_param_attr_sd_b_inner_ptr_nn_dest(int *_Nonnull*_Nonnull); 
// CHECK: void f_on_param_attr_sd_b_inner_ptr_nn_dest(int * __bidi_indexable*);
void f_on_param_attr_sd_b_inner_ptr_nn_dest(int *_Nonnull*_Nonnull);
void f_on_param_attr_sd_b_inner_ptr_nn_dest(int * __bidi_indexable*);

// TODO(dliew): A fix-it should be applied here (rdar://116016096).
// expected-error@+4{{conflicting types for 'f_on_param_attr_sd_b_inner_ptr_nn_src'}}
// CHECK: void f_on_param_attr_sd_b_inner_ptr_nn_src(int **);
// CHECK: void f_on_param_attr_sd_b_inner_ptr_nn_src(int *_Nonnull __bidi_indexable*_Nonnull);
void f_on_param_attr_sd_b_inner_ptr_nn_src(int **);
void f_on_param_attr_sd_b_inner_ptr_nn_src(int *_Nonnull __bidi_indexable*_Nonnull);

// expected-error@+4{{conflicting '__counted_by' attribute with the previous function declaration}}
// CHECK: int *_Nonnull*_Nonnull __counted_by(len) f_on_ret_attr_fd_cb_outer_ptr_nn_dest_nn_src(unsigned len);
// CHECK: int *_Nonnull*_Nonnull __counted_by(len) f_on_ret_attr_fd_cb_outer_ptr_nn_dest_nn_src(unsigned len);
int *_Nonnull*_Nonnull __counted_by(len) f_on_ret_attr_fd_cb_outer_ptr_nn_dest_nn_src(unsigned len);
int *_Nonnull*_Nonnull f_on_ret_attr_fd_cb_outer_ptr_nn_dest_nn_src(unsigned len);

// expected-error@+4{{conflicting '__counted_by' attribute with the previous function declaration}}
// CHECK: int ** __counted_by(len) f_on_ret_attr_fd_cb_outer_ptr_nn_dest(unsigned len);
// CHECK: int *_Nonnull*_Nonnull __counted_by(len) f_on_ret_attr_fd_cb_outer_ptr_nn_dest(unsigned len);
int ** __counted_by(len) f_on_ret_attr_fd_cb_outer_ptr_nn_dest(unsigned len);
int *_Nonnull*_Nonnull f_on_ret_attr_fd_cb_outer_ptr_nn_dest(unsigned len);

// expected-error@+4{{conflicting '__counted_by' attribute with the previous function declaration}}
// CHECK: int *_Nonnull*_Nonnull __counted_by(len) f_on_ret_attr_fd_cb_outer_ptr_nn_src(unsigned len);
// CHECK: int ** __counted_by(len) f_on_ret_attr_fd_cb_outer_ptr_nn_src(unsigned len);
int *_Nonnull*_Nonnull __counted_by(len) f_on_ret_attr_fd_cb_outer_ptr_nn_src(unsigned len);
int ** f_on_ret_attr_fd_cb_outer_ptr_nn_src(unsigned len);

// expected-error@+4{{conflicting types for 'f_on_ret_attr_fd_b_outer_ptr_nn_dest_nn_src'}}
// CHECK: int *_Nonnull*_Nonnull __bidi_indexable f_on_ret_attr_fd_b_outer_ptr_nn_dest_nn_src(void);
// CHECK: int *_Nonnull*_Nonnull __bidi_indexable f_on_ret_attr_fd_b_outer_ptr_nn_dest_nn_src(void);
int *_Nonnull*_Nonnull __bidi_indexable f_on_ret_attr_fd_b_outer_ptr_nn_dest_nn_src(void);
int *_Nonnull*_Nonnull f_on_ret_attr_fd_b_outer_ptr_nn_dest_nn_src(void);

// expected-error@+4{{conflicting types for 'f_on_ret_attr_fd_b_outer_ptr_nn_dest'}}
// CHECK: int ** __bidi_indexable f_on_ret_attr_fd_b_outer_ptr_nn_dest(void);
// CHECK: int *_Nonnull*_Nonnull __bidi_indexable f_on_ret_attr_fd_b_outer_ptr_nn_dest(void);
int ** __bidi_indexable f_on_ret_attr_fd_b_outer_ptr_nn_dest(void);
int *_Nonnull*_Nonnull f_on_ret_attr_fd_b_outer_ptr_nn_dest(void);

// expected-error@+4{{conflicting types for 'f_on_ret_attr_fd_b_outer_ptr_nn_src'}}
// CHECK: int *_Nonnull*_Nonnull __bidi_indexable f_on_ret_attr_fd_b_outer_ptr_nn_src(void);
// CHECK: int ** __bidi_indexable f_on_ret_attr_fd_b_outer_ptr_nn_src(void);
int *_Nonnull*_Nonnull __bidi_indexable f_on_ret_attr_fd_b_outer_ptr_nn_src(void);
int ** f_on_ret_attr_fd_b_outer_ptr_nn_src(void);

// expected-error@+4{{conflicting '__counted_by' attribute with the previous function declaration}}
// CHECK: int *_Nonnull*_Nonnull __counted_by(len) f_on_ret_attr_sd_cb_outer_ptr_nn_dest_nn_src(unsigned len);
// CHECK: int *_Nonnull*_Nonnull __counted_by(len) f_on_ret_attr_sd_cb_outer_ptr_nn_dest_nn_src(unsigned len);
int *_Nonnull*_Nonnull f_on_ret_attr_sd_cb_outer_ptr_nn_dest_nn_src(unsigned len);
int *_Nonnull*_Nonnull __counted_by(len) f_on_ret_attr_sd_cb_outer_ptr_nn_dest_nn_src(unsigned len);

// expected-error@+4{{conflicting '__counted_by' attribute with the previous function declaration}}
// CHECK: int *_Nonnull*_Nonnull __counted_by(len) f_on_ret_attr_sd_cb_outer_ptr_nn_dest(unsigned len);
// CHECK: int ** __counted_by(len) f_on_ret_attr_sd_cb_outer_ptr_nn_dest(unsigned len);
int *_Nonnull*_Nonnull f_on_ret_attr_sd_cb_outer_ptr_nn_dest(unsigned len);
int ** __counted_by(len) f_on_ret_attr_sd_cb_outer_ptr_nn_dest(unsigned len);

// expected-error@+4{{conflicting '__counted_by' attribute with the previous function declaration}}
// CHECK: int ** __counted_by(len) f_on_ret_attr_sd_cb_outer_ptr_nn_src(unsigned len);
// CHECK: int *_Nonnull*_Nonnull __counted_by(len) f_on_ret_attr_sd_cb_outer_ptr_nn_src(unsigned len);
int ** f_on_ret_attr_sd_cb_outer_ptr_nn_src(unsigned len);
int *_Nonnull*_Nonnull __counted_by(len) f_on_ret_attr_sd_cb_outer_ptr_nn_src(unsigned len);

// expected-error@+4{{conflicting types for 'f_on_ret_attr_sd_b_outer_ptr_nn_dest_nn_src'}}
// CHECK: int *_Nonnull*_Nonnull __bidi_indexable f_on_ret_attr_sd_b_outer_ptr_nn_dest_nn_src(void);
// CHECK: int *_Nonnull*_Nonnull __bidi_indexable f_on_ret_attr_sd_b_outer_ptr_nn_dest_nn_src(void);
int *_Nonnull*_Nonnull f_on_ret_attr_sd_b_outer_ptr_nn_dest_nn_src(void);
int *_Nonnull*_Nonnull __bidi_indexable f_on_ret_attr_sd_b_outer_ptr_nn_dest_nn_src(void);

// expected-error@+4{{conflicting types for 'f_on_ret_attr_sd_b_outer_ptr_nn_dest'}}
// CHECK: int *_Nonnull*_Nonnull __bidi_indexable f_on_ret_attr_sd_b_outer_ptr_nn_dest(void);
// CHECK: int ** __bidi_indexable f_on_ret_attr_sd_b_outer_ptr_nn_dest(void);
int *_Nonnull*_Nonnull f_on_ret_attr_sd_b_outer_ptr_nn_dest(void);
int ** __bidi_indexable f_on_ret_attr_sd_b_outer_ptr_nn_dest(void);

// expected-error@+4{{conflicting types for 'f_on_ret_attr_sd_b_outer_ptr_nn_src'}}
// CHECK: int ** __bidi_indexable f_on_ret_attr_sd_b_outer_ptr_nn_src(void);
// CHECK: int *_Nonnull*_Nonnull __bidi_indexable f_on_ret_attr_sd_b_outer_ptr_nn_src(void);
int ** f_on_ret_attr_sd_b_outer_ptr_nn_src(void);
int *_Nonnull*_Nonnull __bidi_indexable f_on_ret_attr_sd_b_outer_ptr_nn_src(void);

// TODO(dliew): A fix-it should be applied here (rdar://116016096).
// expected-error@+4{{conflicting types for 'f_on_ret_attr_fd_b_inner_ptr_nn_dest_nn_src'}}
// CHECK: int *_Nonnull __bidi_indexable*_Nonnull f_on_ret_attr_fd_b_inner_ptr_nn_dest_nn_src(void);
// CHECK: int *_Nonnull*_Nonnull f_on_ret_attr_fd_b_inner_ptr_nn_dest_nn_src(void);
int *_Nonnull __bidi_indexable*_Nonnull f_on_ret_attr_fd_b_inner_ptr_nn_dest_nn_src(void);
int *_Nonnull*_Nonnull f_on_ret_attr_fd_b_inner_ptr_nn_dest_nn_src(void);

// TODO(dliew): A fix-it should be applied here (rdar://116016096).
// expected-error@+4{{conflicting types for 'f_on_ret_attr_fd_b_inner_ptr_nn_dest'}}
// CHECK: int * __bidi_indexable* f_on_ret_attr_fd_b_inner_ptr_nn_dest(void);
// CHECK: int *_Nonnull*_Nonnull f_on_ret_attr_fd_b_inner_ptr_nn_dest(void);
int * __bidi_indexable* f_on_ret_attr_fd_b_inner_ptr_nn_dest(void);
int *_Nonnull*_Nonnull f_on_ret_attr_fd_b_inner_ptr_nn_dest(void);

// TODO(dliew): A fix-it should be applied here (rdar://116016096).
// expected-error@+4{{conflicting types for 'f_on_ret_attr_fd_b_inner_ptr_nn_src'}}
// CHECK: int *_Nonnull __bidi_indexable*_Nonnull f_on_ret_attr_fd_b_inner_ptr_nn_src(void);
// CHECK: int ** f_on_ret_attr_fd_b_inner_ptr_nn_src(void);
int *_Nonnull __bidi_indexable*_Nonnull f_on_ret_attr_fd_b_inner_ptr_nn_src(void);
int ** f_on_ret_attr_fd_b_inner_ptr_nn_src(void);

// TODO(dliew): A fix-it should be applied here (rdar://116016096).
// expected-error@+4{{conflicting types for 'f_on_ret_attr_sd_b_inner_ptr_nn_dest_nn_src'}}
// CHECK: int *_Nonnull*_Nonnull f_on_ret_attr_sd_b_inner_ptr_nn_dest_nn_src(void);
// CHECK: int *_Nonnull __bidi_indexable*_Nonnull f_on_ret_attr_sd_b_inner_ptr_nn_dest_nn_src(void);
int *_Nonnull*_Nonnull f_on_ret_attr_sd_b_inner_ptr_nn_dest_nn_src(void);
int *_Nonnull __bidi_indexable*_Nonnull f_on_ret_attr_sd_b_inner_ptr_nn_dest_nn_src(void);

// TODO(dliew): A fix-it should be applied here (rdar://116016096).
// expected-error@+4{{conflicting types for 'f_on_ret_attr_sd_b_inner_ptr_nn_dest'}}
// CHECK: int *_Nonnull*_Nonnull f_on_ret_attr_sd_b_inner_ptr_nn_dest(void);
// CHECK: int * __bidi_indexable* f_on_ret_attr_sd_b_inner_ptr_nn_dest(void);
int *_Nonnull*_Nonnull f_on_ret_attr_sd_b_inner_ptr_nn_dest(void);
int * __bidi_indexable* f_on_ret_attr_sd_b_inner_ptr_nn_dest(void);

// TODO(dliew): A fix-it should be applied here (rdar://116016096).
// expected-error@+4{{conflicting types for 'f_on_ret_attr_sd_b_inner_ptr_nn_src'}}
// CHECK: int ** f_on_ret_attr_sd_b_inner_ptr_nn_src(void);
// CHECK: int *_Nonnull __bidi_indexable*_Nonnull f_on_ret_attr_sd_b_inner_ptr_nn_src(void);
int ** f_on_ret_attr_sd_b_inner_ptr_nn_src(void);
int *_Nonnull __bidi_indexable*_Nonnull f_on_ret_attr_sd_b_inner_ptr_nn_src(void);
