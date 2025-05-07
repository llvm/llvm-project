// TODO: We should get the same diagnostics with/without return_size (rdar://138982703)

// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -verify=expected %s
// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -verify=expected,rs -fbounds-safety-bringup-missing-checks=return_size %s
// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc -verify=expected %s
// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc -verify=expected,rs -fbounds-safety-bringup-missing-checks=return_size %s

#include <ptrcheck.h>
#include <stddef.h>

void const_count_callee(int *__counted_by(10) ccp);

int *__counted_by(10) const_count(void) {
  // expected-error@+1{{initializing 'cc' of type 'int *__single __counted_by(10)' (aka 'int *__single') and count value of 10 with null always fails}}
  int *__counted_by(10) cc = NULL;

  // expected-error@+1{{assigning null to 'cc' of type 'int *__single __counted_by(10)' (aka 'int *__single') with count value of 10 always fails}}
  cc = NULL;

  // expected-error@+1{{passing null to parameter 'ccp' of type 'int *__single __counted_by(10)' (aka 'int *__single') with count value of 10 always fails}}
  const_count_callee(NULL);

  // rs-error@+1{{returning null from a function with result type 'int *__single __counted_by(10)' (aka 'int *__single') and count value of 10 always fails}}
  return NULL;
}

void const_size_callee(char *__sized_by(10) csp);

char *__sized_by(10) const_size(void) {
  // expected-error@+1{{initializing 'cs' of type 'char *__single __sized_by(10)' (aka 'char *__single') and size value of 10 with null always fails}}
  char *__sized_by(10) cs = NULL;

  // expected-error@+1{{assigning null to 'cs' of type 'char *__single __sized_by(10)' (aka 'char *__single') with size value of 10 always fails}}
  cs = NULL;

  // expected-error@+1{{passing null to parameter 'csp' of type 'char *__single __sized_by(10)' (aka 'char *__single') with size value of 10 always fails}}
  const_size_callee(NULL);

  // rs-error@+1{{returning null from a function with result type 'char *__single __sized_by(10)' (aka 'char *__single') and size value of 10 always fails}}
  return NULL;
}

void dynamic_count_callee(int *__counted_by(len) dcp, int len);

void dynamic_count(void) {
  // expected-error@+2{{initializing 'dc' of type 'int *__single __counted_by(len)' (aka 'int *__single') and count value of 10 with null always fails}}
  int len = 10;
  int *__counted_by(len) dc = NULL;

  // expected-error@+2{{assigning null to 'dc' of type 'int *__single __counted_by(len)' (aka 'int *__single') with count value of 10 always fails}}
  len = 10;
  dc = NULL;

  // expected-error@+1{{passing null to parameter 'dcp' of type 'int *__single __counted_by(len)' (aka 'int *__single') with count value of 10 always fails}}
  dynamic_count_callee(NULL, 10);
}

void dynamic_size_callee(char *__sized_by(size) dsp, int size);

void dynamic_size(void) {
  // expected-error@+2{{initializing 'ds' of type 'char *__single __sized_by(size)' (aka 'char *__single') and size value of 10 with null always fails}}
  int size = 10;
  char *__sized_by(size) ds = NULL;

  // expected-error@+2{{assigning null to 'ds' of type 'char *__single __sized_by(size)' (aka 'char *__single') with size value of 10 always fails}}
  size = 10;
  ds = NULL;

  // expected-error@+1{{passing null to parameter 'dsp' of type 'char *__single __sized_by(size)' (aka 'char *__single') with size value of 10 always fails}}
  dynamic_size_callee(NULL, 10);
}
