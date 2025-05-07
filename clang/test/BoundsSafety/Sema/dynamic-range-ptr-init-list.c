
// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -verify %s
// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc -verify %s

#include <ptrcheck.h>

struct S {
  char *__ended_by(iter) start;
  char *__ended_by(end) iter;
  char *end;
};

void Test(void) {
  char arr[10];
  struct S s_order_all = {arr, arr, arr + 10};
  struct S s_implicit_all;
  // expected-warning@+1{{implicitly initializing field 'iter' of type 'char *__single __ended_by(end) /* __started_by(start) */ ' (aka 'char *__single') to NULL while 'start' is initialized with a value rarely succeeds}}
  struct S s_implicit_partial1 = { arr };
  // expected-warning@+1{{implicitly initializing field 'end' of type 'char *__single /* __started_by(iter) */ ' (aka 'char *__single') to NULL while 'iter' is initialized with a value rarely succeeds}}
  struct S s_implicit_partial2 = { arr, arr+1 };
  // expected-warning@+1{{implicitly initializing field 'iter' of type 'char *__single __ended_by(end) /* __started_by(start) */ ' (aka 'char *__single') to NULL while 'start' is initialized with a value rarely succeeds}}
  struct S s_implicit_partial3 = { .start = arr };
  // expected-warning@+2{{implicitly initializing field 'start' of type 'char *__single __ended_by(iter)' (aka 'char *__single') to NULL while 'iter' is initialized with a value rarely succeeds}}
  // expected-warning@+1{{implicitly initializing field 'end' of type 'char *__single /* __started_by(iter) */ ' (aka 'char *__single') to NULL while 'iter' is initialized with a value rarely succeeds}}
  struct S s_implicit_partial4 = { .iter = arr };
  // expected-warning@+1{{implicitly initializing field 'iter' of type 'char *__single __ended_by(end) /* __started_by(start) */ ' (aka 'char *__single') to NULL while 'end' is initialized with a value rarely succeeds}}
  struct S s_implicit_partial5 = { .end = arr };
  // expected-warning@+2{{implicitly initializing field 'iter' of type 'char *__single __ended_by(end) /* __started_by(start) */ ' (aka 'char *__single') to NULL while 'end' is initialized with a value rarely succeeds}}
  // expected-warning@+1{{implicitly initializing field 'iter' of type 'char *__single __ended_by(end) /* __started_by(start) */ ' (aka 'char *__single') to NULL while 'start' is initialized with a value rarely succeeds}}
  struct S s_implicit_partial6 = { .end = arr, .start = arr };
  // expected-warning@+1{{implicitly initializing field 'end' of type 'char *__single /* __started_by(iter) */ ' (aka 'char *__single') to NULL while 'iter' is initialized with a value rarely succeeds}}
  struct S s_implicit_partial7 = { .iter = arr, .start = arr };
  // expected-warning@+1{{implicitly initializing field 'start' of type 'char *__single __ended_by(iter)' (aka 'char *__single') to NULL while 'iter' is initialized with a value rarely succeeds}}
  struct S s_implicit_partial8 = { .iter = arr, .end = arr };
  // expected-warning@+1{{initializing field 'iter' of type 'char *__single __ended_by(end) /* __started_by(start) */ ' (aka 'char *__single') to NULL while 'start' is initialized with a value rarely succeeds}}
  struct S s_partial1 = { arr, 0, 0 };
  // expected-warning@+1{{initializing field 'end' of type 'char *__single /* __started_by(iter) */ ' (aka 'char *__single') to NULL while 'iter' is initialized with a value rarely succeeds}}
  struct S s_partial2 = { arr, arr+1, 0 };
  // expected-warning@+1{{initializing field 'iter' of type 'char *__single __ended_by(end) /* __started_by(start) */ ' (aka 'char *__single') to NULL while 'start' is initialized with a value rarely succeeds}}
  struct S s_partial3 = { .start = arr, .iter = 0, .end = 0 };
  // expected-warning@+2{{initializing field 'start' of type 'char *__single __ended_by(iter)' (aka 'char *__single') to NULL while 'iter' is initialized with a value rarely succeeds}}
  // expected-warning@+1{{initializing field 'end' of type 'char *__single /* __started_by(iter) */ ' (aka 'char *__single') to NULL while 'iter' is initialized with a value rarely succeeds}}
  struct S s_partial4 = { .iter = arr, .end = 0, .start = 0 };
  // expected-warning@+1{{initializing field 'iter' of type 'char *__single __ended_by(end) /* __started_by(start) */ ' (aka 'char *__single') to NULL while 'end' is initialized with a value rarely succeeds}}
  struct S s_partial5 = { .end = arr, .start = 0, .iter = 0 };
  // expected-warning@+2{{initializing field 'iter' of type 'char *__single __ended_by(end) /* __started_by(start) */ ' (aka 'char *__single') to NULL while 'end' is initialized with a value rarely succeeds}}
  // expected-warning@+1{{initializing field 'iter' of type 'char *__single __ended_by(end) /* __started_by(start) */ ' (aka 'char *__single') to NULL while 'start' is initialized with a value rarely succeeds}}
  struct S s_partial6 = { .end = arr, .start = arr, .iter = 0 };
  // expected-warning@+1{{initializing field 'end' of type 'char *__single /* __started_by(iter) */ ' (aka 'char *__single') to NULL while 'iter' is initialized with a value rarely succeeds}}
  struct S s_partial7 = { .iter = arr, .start = arr, .end = 0 };
  // expected-warning@+1{{initializing field 'start' of type 'char *__single __ended_by(iter)' (aka 'char *__single') to NULL while 'iter' is initialized with a value rarely succeeds}}
  struct S s_partial8 = { .iter = arr, .end = arr, .start = 0 };
  struct S s_designate_all = { .iter = &arr[1], .end = arr + 2, .start = arr };
}
