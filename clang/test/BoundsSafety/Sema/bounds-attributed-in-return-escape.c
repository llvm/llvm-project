

// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -verify %s
// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc -verify %s

#include <ptrcheck.h>

int *__counted_by(n) cb_in(int n) {
  int *n1 = &n; // expected-error{{variable referred to by '__counted_by' cannot be pointed to by any other variable; exception is when the pointer is passed as a compatible argument to a function}}
  int n2 = n;
}

int *__counted_by(*n) cb_out(int *n) {
  int **n1 = &n; // expected-error{{variable referred to by '__counted_by' cannot be pointed to by any other variable; exception is when the pointer is passed as a compatible argument to a function}}
  int *n2 = n;   // expected-error{{variable referred to by '__counted_by' cannot be pointed to by any other variable; exception is when the pointer is passed as a compatible argument to a function}}
  int n3 = *n;
}

void *__sized_by(n) sb_in(int n) {
  int *n1 = &n; // expected-error{{variable referred to by '__sized_by' cannot be pointed to by any other variable; exception is when the pointer is passed as a compatible argument to a function}}
  int n2 = n;
}

int *__counted_by_or_null(n) cbn_in(int n) {
  int *n1 = &n; // expected-error{{variable referred to by '__counted_by_or_null' cannot be pointed to by any other variable; exception is when the pointer is passed as a compatible argument to a function}}
  int n2 = n;
}

void *__sized_by_or_null(n) sbn_in(int n) {
  int *n1 = &n; // expected-error{{variable referred to by '__sized_by_or_null' cannot be pointed to by any other variable; exception is when the pointer is passed as a compatible argument to a function}}
  int n2 = n;
}

int *__ended_by(end) eb_in(int *end) {
  int **e1 = &end; // expected-error{{variable referred to by '__ended_by' cannot be pointed to by any other variable; exception is when the pointer is passed as a compatible argument to a function}}
  int *e2 = end;
  int e3 = *end;
}

int *__ended_by(*end) eb_out(int **end) {
  int ***e1 = &end; // expected-error{{variable referred to by '__ended_by' cannot be pointed to by any other variable; exception is when the pointer is passed as a compatible argument to a function}}
  int **e2 = end;   // expected-error{{variable referred to by '__ended_by' cannot be pointed to by any other variable; exception is when the pointer is passed as a compatible argument to a function}}
  int *e3 = *end;
  int e4 = **end;
}
