// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -verify=expected,rs,compound-literal %s
// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc -verify=expected,rs,compound-literal %s

#include <ptrcheck.h>
#include <stddef.h>

struct cb {
  int* __counted_by(10) ptr;
};

void const_count_callee(int *__counted_by(10) ccp);

int *__counted_by(10) const_count(void) {
  // expected-error@+1{{initializing 'cc' of type 'int *__single __counted_by(10)' (aka 'int *__single') and count value of 10 with null always fails}}
  int *__counted_by(10) cc = NULL;

  // expected-error@+1{{assigning null to 'cc' of type 'int *__single __counted_by(10)' (aka 'int *__single') with count value of 10 always fails}}
  cc = NULL;

  // expected-error@+1{{passing null to parameter 'ccp' of type 'int *__single __counted_by(10)' (aka 'int *__single') with count value of 10 always fails}}
  const_count_callee(NULL);

  // expected-error@+1{{initializing 'local_cb.ptr' of type 'int *__single __counted_by(10)' (aka 'int *__single') and count value of 10 with null always fails}}
  struct cb local_cb = { NULL };
  // compound-literal-error@+1{{initializing 'int *__single __counted_by(10)' (aka 'int *__single') and count value of 10 with null always fails}}
  local_cb = (struct cb){ NULL };

  // rs-error@+1{{returning null from a function with result type 'int *__single __counted_by(10)' (aka 'int *__single') and count value of 10 always fails}}
  return NULL;
}

int *__counted_by(10) const_count_null_cast(void) {
  // expected-error@+1{{initializing 'cc' of type 'int *__single __counted_by(10)' (aka 'int *__single') and count value of 10 with null always fails}}
  int *__counted_by(10) cc = (int*) NULL;

  // expected-error@+1{{assigning null to 'cc' of type 'int *__single __counted_by(10)' (aka 'int *__single') with count value of 10 always fails}}
  cc = (int*) NULL;

  // expected-error@+1{{passing null to parameter 'ccp' of type 'int *__single __counted_by(10)' (aka 'int *__single') with count value of 10 always fails}}
  const_count_callee((int*) NULL);

  // expected-error@+1{{initializing 'local_cb.ptr' of type 'int *__single __counted_by(10)' (aka 'int *__single') and count value of 10 with null always fails}}
  struct cb local_cb = { (int*) NULL };
  // compound-literal-error@+1{{initializing 'int *__single __counted_by(10)' (aka 'int *__single') and count value of 10 with null always fails}}
  local_cb = (struct cb){ (int*) NULL };

  // rs-error@+1{{returning null from a function with result type 'int *__single __counted_by(10)' (aka 'int *__single') and count value of 10 always fails}}
  return (int*) NULL;
}

struct sb {
  char* __sized_by(10) ptr;
};

void const_size_callee(char *__sized_by(10) csp);

char *__sized_by(10) const_size(void) {
  // expected-error@+1{{initializing 'cs' of type 'char *__single __sized_by(10)' (aka 'char *__single') and size value of 10 with null always fails}}
  char *__sized_by(10) cs = NULL;

  // expected-error@+1{{assigning null to 'cs' of type 'char *__single __sized_by(10)' (aka 'char *__single') with size value of 10 always fails}}
  cs = NULL;

  // expected-error@+1{{passing null to parameter 'csp' of type 'char *__single __sized_by(10)' (aka 'char *__single') with size value of 10 always fails}}
  const_size_callee(NULL);

  // expected-error@+1{{initializing 'local_sb.ptr' of type 'char *__single __sized_by(10)' (aka 'char *__single') and size value of 10 with null always fails}}
  struct sb local_sb = { NULL };
  // compound-literal-error@+1{{initializing 'char *__single __sized_by(10)' (aka 'char *__single') and size value of 10 with null always fails}}
  local_sb = (struct sb){ NULL };

  // rs-error@+1{{returning null from a function with result type 'char *__single __sized_by(10)' (aka 'char *__single') and size value of 10 always fails}}
  return NULL;
}

char *__sized_by(10) const_size_null_cast(void) {
  // expected-error@+1{{initializing 'cs' of type 'char *__single __sized_by(10)' (aka 'char *__single') and size value of 10 with null always fails}}
  char *__sized_by(10) cs = (char*) NULL;

  // expected-error@+1{{assigning null to 'cs' of type 'char *__single __sized_by(10)' (aka 'char *__single') with size value of 10 always fails}}
  cs = (char*) NULL;

  // expected-error@+1{{passing null to parameter 'csp' of type 'char *__single __sized_by(10)' (aka 'char *__single') with size value of 10 always fails}}
  const_size_callee((char*) NULL);

  // expected-error@+1{{initializing 'local_sb.ptr' of type 'char *__single __sized_by(10)' (aka 'char *__single') and size value of 10 with null always fails}}
  struct sb local_sb = { (char*) NULL };
  // compound-literal-error@+1{{initializing 'char *__single __sized_by(10)' (aka 'char *__single') and size value of 10 with null always fails}}
  local_sb = (struct sb){ (char*) NULL };

  // rs-error@+1{{returning null from a function with result type 'char *__single __sized_by(10)' (aka 'char *__single') and size value of 10 always fails}}
  return (char*) NULL;
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

void dynamic_count_null_cast(void) {
  // expected-error@+2{{initializing 'dc' of type 'int *__single __counted_by(len)' (aka 'int *__single') and count value of 10 with null always fails}}
  int len = 10;
  int *__counted_by(len) dc = (int*) NULL;

  // expected-error@+2{{assigning null to 'dc' of type 'int *__single __counted_by(len)' (aka 'int *__single') with count value of 10 always fails}}
  len = 10;
  dc = (int*) NULL;

  // expected-error@+1{{passing null to parameter 'dcp' of type 'int *__single __counted_by(len)' (aka 'int *__single') with count value of 10 always fails}}
  dynamic_count_callee((int*) NULL, 10);
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

void dynamic_size_null_cast(void) {
  // expected-error@+2{{initializing 'ds' of type 'char *__single __sized_by(size)' (aka 'char *__single') and size value of 10 with null always fails}}
  int size = 10;
  char *__sized_by(size) ds = (char*) NULL;

  // expected-error@+2{{assigning null to 'ds' of type 'char *__single __sized_by(size)' (aka 'char *__single') with size value of 10 always fails}}
  size = 10;
  ds = (char*) NULL;

  // expected-error@+1{{passing null to parameter 'dsp' of type 'char *__single __sized_by(size)' (aka 'char *__single') with size value of 10 always fails}}
  dynamic_size_callee((char*) NULL, 10);
}
