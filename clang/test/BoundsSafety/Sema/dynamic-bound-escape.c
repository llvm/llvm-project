

// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -verify %s
// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc -verify %s

#include <ptrcheck.h>

// __counted_by

void cb_in_in(int *__counted_by(len) buf, int len) {
  int **b1 = &buf; // expected-error{{pointer with '__counted_by' cannot be pointed to by any other variable; exception is when the variable is passed as a compatible argument to a function}}
  int *b2 = buf;
  int b3 = *buf;

  int *l1 = &len; // expected-error{{variable referred to by '__counted_by' cannot be pointed to by any other variable; exception is when the pointer is passed as a compatible argument to a function}}
  int l2 = len;
}

void cb_in_out(int *__counted_by(*len) buf, int *len) {
  int **b1 = &buf; // expected-error{{pointer with '__counted_by' cannot be pointed to by any other variable; exception is when the variable is passed as a compatible argument to a function}}
  int *b2 = buf;
  int b3 = *buf;

  int **l1 = &len; // expected-error{{variable referred to by '__counted_by' cannot be pointed to by any other variable; exception is when the pointer is passed as a compatible argument to a function}}
  int *l2 = len;   // expected-error{{variable referred to by '__counted_by' cannot be pointed to by any other variable; exception is when the pointer is passed as a compatible argument to a function}}
  int l3 = *len;
}

void cb_out_in(int *__counted_by(len) * buf, int len) {
  int ***b1 = &buf; // expected-error{{pointer with '__counted_by' cannot be pointed to by any other variable; exception is when the variable is passed as a compatible argument to a function}}
  int **b2 = buf;   // expected-error{{pointer with '__counted_by' cannot be pointed to by any other variable; exception is when the variable is passed as a compatible argument to a function}}
  int *b3 = *buf;
  int b4 = **buf;

  int *l1 = &len; // expected-error{{variable referred to by '__counted_by' cannot be pointed to by any other variable; exception is when the pointer is passed as a compatible argument to a function}}
  int l2 = len;
}

void cb_out_out(int *__counted_by(*len) * buf, int *len) {
  int ***b1 = &buf; // expected-error{{pointer with '__counted_by' cannot be pointed to by any other variable; exception is when the variable is passed as a compatible argument to a function}}
  int **b2 = buf;   // expected-error{{pointer with '__counted_by' cannot be pointed to by any other variable; exception is when the variable is passed as a compatible argument to a function}}
  int *b3 = *buf;
  int b4 = **buf;

  int **l1 = &len; // expected-error{{variable referred to by '__counted_by' cannot be pointed to by any other variable; exception is when the pointer is passed as a compatible argument to a function}}
  int *l2 = len;   // expected-error{{variable referred to by '__counted_by' cannot be pointed to by any other variable; exception is when the pointer is passed as a compatible argument to a function}}
  int l3 = *len;
}

void ptr_cb_in_in(int **__counted_by(len) buf, int len) {
  int ***b1 = &buf; // expected-error{{pointer with '__counted_by' cannot be pointed to by any other variable; exception is when the variable is passed as a compatible argument to a function}}
  int **b2 = buf;
  int *b3 = *buf;
  int b4 = **buf;
}

void ptr_cb_out_in(int **__counted_by(len) * buf, int len) {
  int ****b1 = &buf; // expected-error{{pointer with '__counted_by' cannot be pointed to by any other variable; exception is when the variable is passed as a compatible argument to a function}}
  int ***b2 = buf;   // expected-error{{pointer with '__counted_by' cannot be pointed to by any other variable; exception is when the variable is passed as a compatible argument to a function}}
  int **b3 = *buf;
  int *b4 = **buf;
  int b5 = ***buf;
}

// __sized_by
// Just check if the diagnostic says __sized_by instead of __counted_by.

void sb_in_in(int *__sized_by(size) buf, int size) {
  int **b1 = &buf; // expected-error{{pointer with '__sized_by' cannot be pointed to by any other variable; exception is when the variable is passed as a compatible argument to a function}}
  int *b2 = buf;
  int b3 = *buf;

  int *s1 = &size; // expected-error{{variable referred to by '__sized_by' cannot be pointed to by any other variable; exception is when the pointer is passed as a compatible argument to a function}}
  int s2 = size;
}

// __ended_by

void eb_in_in(int *__ended_by(end) buf, int *end) {
  int **b1 = &buf; // expected-error{{pointer with '__ended_by' cannot be pointed to by any other variable; exception is when the variable is passed as a compatible argument to a function}}
  int *b2 = buf;
  int b3 = *buf;

  int **e1 = &end; // expected-error{{variable referred to by '__ended_by' cannot be pointed to by any other variable; exception is when the pointer is passed as a compatible argument to a function}}
  int *e2 = end;
  int e3 = *end;
}

void eb_in_out(int *__ended_by(*end) buf, int **end) {
  int **b1 = &buf; // expected-error{{pointer with '__ended_by' cannot be pointed to by any other variable; exception is when the variable is passed as a compatible argument to a function}}
  int *b2 = buf;
  int b3 = *buf;

  int ***e1 = &end; // expected-error{{variable referred to by '__ended_by' cannot be pointed to by any other variable; exception is when the pointer is passed as a compatible argument to a function}}
  int **e2 = end;   // expected-error{{variable referred to by '__ended_by' cannot be pointed to by any other variable; exception is when the pointer is passed as a compatible argument to a function}}
  int *e3 = *end;
  int e4 = **end;
}

void eb_out_in(int *__ended_by(end) * buf, int *end) {
  int ***b1 = &buf; // expected-error{{pointer with '__ended_by' cannot be pointed to by any other variable; exception is when the variable is passed as a compatible argument to a function}}
  int **b2 = buf;   // expected-error{{pointer with '__ended_by' cannot be pointed to by any other variable; exception is when the variable is passed as a compatible argument to a function}}
  int *b3 = *buf;
  int b4 = **buf;

  int **e1 = &end; // expected-error{{variable referred to by '__ended_by' cannot be pointed to by any other variable; exception is when the pointer is passed as a compatible argument to a function}}
  int *e2 = end;
  int e3 = *end;
}

void eb_out_out(int *__ended_by(*end) * buf, int **end) {
  int ***b1 = &buf; // expected-error{{pointer with '__ended_by' cannot be pointed to by any other variable; exception is when the variable is passed as a compatible argument to a function}}
  int **b2 = buf;   // expected-error{{pointer with '__ended_by' cannot be pointed to by any other variable; exception is when the variable is passed as a compatible argument to a function}}
  int *b3 = *buf;
  int b4 = **buf;

  int ***e1 = &end; // expected-error{{variable referred to by '__ended_by' cannot be pointed to by any other variable; exception is when the pointer is passed as a compatible argument to a function}}
  int **e2 = end;   // expected-error{{variable referred to by '__ended_by' cannot be pointed to by any other variable; exception is when the pointer is passed as a compatible argument to a function}}
  int *e3 = *end;
  int e4 = **end;
}

void ptr_eb_in_in(int **__ended_by(end) buf, int **end) {
  int ***b1 = &buf; // expected-error{{pointer with '__ended_by' cannot be pointed to by any other variable; exception is when the variable is passed as a compatible argument to a function}}
  int **b2 = buf;
  int *b3 = *buf;
  int b4 = **buf;

  int ***e1 = &end; // expected-error{{variable referred to by '__ended_by' cannot be pointed to by any other variable; exception is when the pointer is passed as a compatible argument to a function}}
  int **e2 = end;
  int *e3 = *end;
  int e4 = **end;
}

void ptr_eb_out_out(int **__ended_by(*end) * buf, int ***end) {
  int ****b1 = &buf; // expected-error{{pointer with '__ended_by' cannot be pointed to by any other variable; exception is when the variable is passed as a compatible argument to a function}}
  int ***b2 = buf;   // expected-error{{pointer with '__ended_by' cannot be pointed to by any other variable; exception is when the variable is passed as a compatible argument to a function}}
  int **b3 = *buf;
  int *b4 = **buf;
  int b5 = ***buf;

  int ****e1 = &end; // expected-error{{variable referred to by '__ended_by' cannot be pointed to by any other variable; exception is when the pointer is passed as a compatible argument to a function}}
  int ***e2 = end;   // expected-error{{variable referred to by '__ended_by' cannot be pointed to by any other variable; exception is when the pointer is passed as a compatible argument to a function}}
  int **e3 = *end;
  int *e4 = **end;
  int e5 = ***end;
}
