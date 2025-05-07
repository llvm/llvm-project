

// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -verify %s
// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc -verify %s
// RUN: %clang_cc1 -fsyntax-only -fexperimental-bounds-safety-attributes -x c -verify %s
// RUN: %clang_cc1 -fsyntax-only -fexperimental-bounds-safety-attributes -x c++ -verify %s
// RUN: %clang_cc1 -fsyntax-only -fexperimental-bounds-safety-attributes -x objective-c -verify %s
// RUN: %clang_cc1 -fsyntax-only -fexperimental-bounds-safety-attributes -x objective-c++ -verify %s

#include <ptrcheck.h>

typedef int *__counted_by(10) giptr_counted_t; // expected-error{{'__counted_by' inside typedef is only allowed for function type}}
typedef int *__sized_by(1) giptr_sized_t;      // expected-error{{'__sized_by' inside typedef is only allowed for function type}}
typedef int *__counted_by_or_null(10) giptr_counted_t; // expected-error{{'__counted_by_or_null' inside typedef is only allowed for function type}}
typedef int *__sized_by_or_null(1) giptr_sized_t;      // expected-error{{'__sized_by_or_null' inside typedef is only allowed for function type}}
typedef int *__counted_by(10) * foo_t;         // expected-error{{'__counted_by' inside typedef is only allowed for function type}}

giptr_counted_t gip_cnt;
giptr_sized_t gip_siz;

void foo() {
  giptr_counted_t gip_cnt_local;

  int n;
  typedef int *__counted_by(n) liptr_counted_t; // expected-error{{'__counted_by' inside typedef is only allowed for function type}}
}

typedef int *__counted_by(16) f1(void);     // ok
typedef void *__sized_by(16) f2(void);      // ok
typedef int *__counted_by(len) f3(int len); // ok
typedef void *__sized_by(len) f4(int len);  // ok

typedef int *__counted_by_or_null(16) f5(void);     // ok
typedef void *__sized_by_or_null(16) f6(void);      // ok
typedef int *__counted_by_or_null(len) f7(int len); // ok
typedef void *__sized_by_or_null(len) f8(int len);  // ok

typedef int *__counted_by(16) (*fp1)(void);     // ok
typedef void *__sized_by(16) (*fp2)(void);      // ok
typedef int *__counted_by(len) (*fp3)(int len); // ok
typedef void *__sized_by(len) (*fp4)(int len);  // ok

typedef int *__counted_by_or_null(16) (*fp5)(void);     // ok
typedef void *__sized_by_or_null(16) (*fp6)(void);      // ok
typedef int *__counted_by_or_null(len) (*fp7)(int len); // ok
typedef void *__sized_by_or_null(len) (*fp8)(int len);  // ok
