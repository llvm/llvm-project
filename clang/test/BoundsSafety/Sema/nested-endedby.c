
// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -verify %s

#include <ptrcheck.h>

struct S {
  unsigned len;
  // expected-error@+1{{'__ended_by' attribute on nested pointer type is only allowed on indirect parameters}}
  int *__ended_by(end) *__counted_by(len) buf;
  int **end;
  // expected-error@+1{{'__ended_by' attribute on nested pointer type is only allowed on indirect parameters}}
  int *__ended_by(end2) *__ended_by(end3) buf2;
  int **end2;
  int *end3;
};

int fooOut(int *__ended_by(end) *out_start, int *end); // ok
int fooOutOut(int *__ended_by(*out_end) *out_buf, int **out_end); // ok
// FIXME: doesn't sound exactly right
// expected-error@+1{{'__ended_by' attribute on nested pointer type is only allowed on indirect parameters}}
int bar(int *__ended_by(end1) *__ended_by(end2) out_buf, int **end1, int *end2);

void baz() {
  int *end;
  int *__ended_by(end) start;

  fooOut(&start, end);
  fooOutOut(&start, &end);

  int **p = &end; // expected-error{{variable referred to by '__ended_by' cannot be pointed to by any other variable}}
  // expected-error@+1{{pointer with '__ended_by' cannot be pointed to by any other variable}}
  p = &start;
}

void qux() {
  int *end;

  {
    // expected-error@+1{{'__ended_by' attribute on nested pointer type is only allowed on indirect parameters}}
    int *__ended_by(end) *nested_buf;
    // expected-error@+2{{local variable buf must be declared right next to its dependent decl}}
    // expected-error@+1{{argument of '__ended_by' attribute cannot refer to declaration from a different scope}}
    int *__ended_by(end) buf;
  }

  end = 0;
}
