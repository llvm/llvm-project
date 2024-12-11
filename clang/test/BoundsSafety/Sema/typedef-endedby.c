// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -verify %s
// RUN: %clang_cc1 -fsyntax-only -fexperimental-bounds-safety-attributes -x c -verify %s
// RUN: %clang_cc1 -fsyntax-only -fexperimental-bounds-safety-attributes -x objective-c -verify %s
// RUN: %clang_cc1 -fsyntax-only -fexperimental-bounds-safety-attributes -x c++ -verify %s
// RUN: %clang_cc1 -fsyntax-only -fexperimental-bounds-safety-attributes -x objective-c++ -verify %s

#include <ptrcheck.h>

int *gend;
// expected-error@+1{{'__ended_by' inside typedef is only allowed for function type}}
typedef int *__ended_by(gend) endedz_t;
// expected-error@+1{{'__ended_by' inside typedef is only allowed for function type}}
typedef int *__ended_by(gend) * endedz_nested_t;


void foo() {
  endedz_t gip_cnt_local;

  int *end;
  typedef int *__ended_by(end) liptr_counted_t; // expected-error{{'__ended_by' inside typedef is only allowed for function type}}
}

typedef int *__ended_by(pend) f(int *pend);     // ok
typedef int *__ended_by(pend) (*fp)(void *pend);// ok
