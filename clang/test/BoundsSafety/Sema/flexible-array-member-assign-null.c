
// RUN: %clang_cc1 -fbounds-safety -verify %s
// RUN: %clang_cc1 -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc -verify %s

#include <ptrcheck.h>

struct flexible {
    int count;
    int elems[__counted_by(count)];
};

void init_null(void) {
  struct flexible *__single s = 0;
}

void init_null_bidi(void) {
  struct flexible *s = 0;
}

void init_casted_null(void) {
  struct flexible *__single s = (struct flexible *)0;
}

void init_casted_null_bidi(void) {
  struct flexible *s = (struct flexible *)0;
}

void assign_null(void) {
  struct flexible *__single s;
  s = 0;
}

void assign_casted_null(void) {
  struct flexible *__single s;
  s = (struct flexible *)0;
}

void impl_init_null_member_ref(void) {
  struct flexible *__single s;
  s->count = 10; // expected-error{{assignment to 's->count' requires an immediately preceding assignment to 's' with a wide pointer}}
}

void init_null_member_ref(void) {
  struct flexible *__single s = 0;
  s->count = 10; // expected-error{{base of member reference is a null pointer}}
}

void init_casted_null_member_ref(void) {
  struct flexible *__single s = (struct flexible *)0;
  s->count = 0; // expected-error{{base of member reference is a null pointer}}
}

void assign_null_member_ref(void) {
  struct flexible *__single s;
  s = 0;
  s->count = 2; // expected-error{{base of member reference is a null pointer}}
}

void assign_casted_null_member_ref(void) {
  struct flexible *__single s;
  s = (struct flexible *)0;
  s->count = 7; // expected-error{{base of member reference is a null pointer}}
}

void assign_casted_null_member_ref_bidi(void) {
  struct flexible *s;
  s = (struct flexible *)0;
  s->count = 7; // XXX: currently, flexible array member operations are analyzed only when their base is a single pointer type
}
