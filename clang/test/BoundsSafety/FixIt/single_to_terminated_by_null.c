
// RUN: cp %s %t
// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -fdiagnostics-parseable-fixits -verify %t
// RUN: not %clang_cc1 -fsyntax-only -fbounds-safety -fdiagnostics-parseable-fixits %t > %t.cc_out 2> %t.cc_out
// RUN: FileCheck %s --input-file=%t.cc_out

#include <ptrcheck.h>

// expected-note@+1 6{{passing argument to parameter 'path' here}}
void method_with_single_const_param(const char *path); 

// expected-note@+1 5{{passing argument to parameter 'path' here}}
void method_with_single_null_terminated_param(char * __null_terminated path);

struct Foo {
// expected-note@+4 {{consider adding '__null_terminated' to 'buf'}}
// CHECK-DAG: fix-it:"{{.+}}single_to_terminated_by_null.c.tmp":{[[@LINE+3]]:9-[[@LINE+3]]:9}:"__null_terminated "
// expected-note@+2{{consider adding 'const' to 'buf'}}
// CHECK-DAG: fix-it:"{{.+}}single_to_terminated_by_null.c.tmp":{[[@LINE+1]]:3-[[@LINE+1]]:3}:"const "
  char *buf;

// expected-note@+4 {{consider adding '__null_terminated' to 'arr_in_struct'}}
// CHECK-DAG: fix-it:"{{.+}}single_to_terminated_by_null.c.tmp":{[[@LINE+3]]:9-[[@LINE+3]]:9}:"__null_terminated "
// expected-note@+2{{consider adding 'const' to 'arr_in_struct'}}
// CHECK-DAG: fix-it:"{{.+}}single_to_terminated_by_null.c.tmp":{[[@LINE+1]]:3-[[@LINE+1]]:3}:"const "
  char *arr_in_struct[4];
};

void test_invocation_with_struct_fields(struct Foo f) {

  // expected-error@+1 {{passing 'char *__single' to parameter of incompatible type 'const char *__single __terminated_by(0)' (aka 'const char *__single') is an unsafe operation; use '__unsafe_null_terminated_from_indexable()' or '__unsafe_forge_null_terminated()' to perform this conversion}}
  method_with_single_const_param(f.arr_in_struct[0]);

   // expected-error@+1 {{passing 'char *__single' to parameter of incompatible type 'char *__single __terminated_by(0)' (aka 'char *__single') is an unsafe operation; use '__unsafe_null_terminated_from_indexable()' or '__unsafe_forge_null_terminated()' to perform this conversion}}
  method_with_single_null_terminated_param(f.arr_in_struct[0]);

   // expected-error@+1 {{passing 'char *__single' to parameter of incompatible type 'const char *__single __terminated_by(0)' (aka 'const char *__single') is an unsafe operation; use '__unsafe_null_terminated_from_indexable()' or '__unsafe_forge_null_terminated()' to perform this conversion}} 
   method_with_single_const_param(f.buf);

   // expected-error@+1 {{passing 'char *__single' to parameter of incompatible type 'char *__single __terminated_by(0)' (aka 'char *__single') is an unsafe operation; use '__unsafe_null_terminated_from_indexable()' or '__unsafe_forge_null_terminated()' to perform this conversion}}
   method_with_single_null_terminated_param(f.buf);
}

// expected-note@+2{{consider adding 'const' to 'anchor'}}
// CHECK-DAG: fix-it:"{{.+}}single_to_terminated_by_null.c.tmp":{[[@LINE+1]]:25-[[@LINE+1]]:25}:"const "
void single_invocation1(char *anchor)
{
  // expected-error@+1 {{passing 'char *__single' to parameter of incompatible type 'const char *__single __terminated_by(0)' (aka 'const char *__single') is an unsafe operation; use '__unsafe_null_terminated_from_indexable()' or '__unsafe_forge_null_terminated()' to perform this conversion}}
  method_with_single_const_param(anchor);
}

// expected-note@+2{{consider adding '__null_terminated' to 'anchor'}}
// CHECK-DAG: fix-it:"{{.+}}single_to_terminated_by_null.c.tmp":{[[@LINE+1]]:31-[[@LINE+1]]:31}:"__null_terminated "
void single_invocation2(char *anchor)
{
  //expected-error@+1 {{passing 'char *__single' to parameter of incompatible type 'char *__single __terminated_by(0)' (aka 'char *__single') is an unsafe operation; use '__unsafe_null_terminated_from_indexable()' or '__unsafe_forge_null_terminated()' to perform this conversion}} 
  method_with_single_null_terminated_param(anchor);
}

void test_single_invocation_local_variable() {
  // expected-note@+2 {{consider adding 'const' to 'local_arr'}}
  // CHECK-DAG: fix-it:"{{.+}}single_to_terminated_by_null.c.tmp":{[[@LINE+1]]:3-[[@LINE+1]]:3}:"const "
  char *local_arr[2];

  // expected-error@+1{{passing 'char *__single' to parameter of incompatible type 'const char *__single __terminated_by(0)' (aka 'const char *__single') is an unsafe operation; use '__unsafe_null_terminated_from_indexable()' or '__unsafe_forge_null_terminated()' to perform this conversion}}
  method_with_single_const_param(local_arr[0]);
}

void test_single_invocation2_local_variable() {
  // expected-note@+2 {{consider adding '__null_terminated' to 'local_arr'}}
  // CHECK-DAG: fix-it:"{{.+}}single_to_terminated_by_null.c.tmp":{[[@LINE+1]]:9-[[@LINE+1]]:9}:"__null_terminated "
  char *local_arr[2];

    //expected-error@+1 {{passing 'char *__single' to parameter of incompatible type 'char *__single __terminated_by(0)' (aka 'char *__single') is an unsafe operation; use '__unsafe_null_terminated_from_indexable()' or '__unsafe_forge_null_terminated()' to perform this conversion}}
  method_with_single_null_terminated_param(local_arr[0]);
}

void test_single_invocation4_local_variable() {
  char * __single local_arr[2];

  // expected-error@+1{{passing 'char *__single' to parameter of incompatible type 'const char *__single __terminated_by(0)' (aka 'const char *__single') is an unsafe operation; use '__unsafe_null_terminated_from_indexable()' or '__unsafe_forge_null_terminated()' to perform this conversion}}
  method_with_single_const_param(local_arr[0]);
}

// TODO: Emit a note and a fixit here. rdar://122997544
char * returns_single(void);

void test_invocation_with_return_values() {

  // expected-error@+1 {{passing 'char *__single' to parameter of incompatible type 'const char *__single __terminated_by(0)' (aka 'const char *__single') is an unsafe operation; use '__unsafe_null_terminated_from_indexable()' or '__unsafe_forge_null_terminated()' to perform this conversion}}
  method_with_single_const_param(returns_single());

  // expected-error@+1 {{passing 'char *__single' to parameter of incompatible type 'char *__single __terminated_by(0)' (aka 'char *__single') is an unsafe operation; use '__unsafe_null_terminated_from_indexable()' or '__unsafe_forge_null_terminated()' to perform this conversion}}
  method_with_single_null_terminated_param(returns_single());
}
