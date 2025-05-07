
// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -verify %s
// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc -verify %s
#include <ptrcheck.h>

void Foo(int *__ended_by(end) start, int *end) {
  start--;      // expected-error{{negative pointer arithmetic on pointer that starts the '__ended_by' chain always traps}}
  ++end;        // expected-error{{positive pointer arithmetic on end pointer always traps}}
}

void Bar(int *__ended_by(end) start, int *end) {
  start+=4;     // expected-error{{assignment to 'int *__single __ended_by(end)' (aka 'int *__single') 'start' requires corresponding assignment to 'end'; add self assignment 'end = end' if the value has not changed}}
}

typedef struct {
  char *__ended_by(iter) start;
  char *__ended_by(end) iter;
  char *end;
} T;

void Baz(T *tp) {
  --tp->start;  // expected-error{{negative pointer arithmetic on pointer that starts the '__ended_by' chain always traps}}
  tp->end++;    // expected-error{{positive pointer arithmetic on end pointer always traps}}
}

void Qux(T *tp) {
  tp->start = tp->start;
  tp->end = tp->end;
  tp->iter--;
}

void Quux(T *tp) {
  tp->start++;
  ++tp->iter;
  --tp->end;
}
