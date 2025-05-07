
// RUN: %clang_cc1 -fbounds-safety -verify %s
// RUN: %clang_cc1 -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc -verify %s

#include <ptrcheck.h>

struct flexible {
    int count;
    int elems[__counted_by(count)];
};

void init_single(void *p) {
  struct flexible *__single s = p;
}

void init_casted_single(void *p) {
  struct flexible *__single s = (struct flexible *)p;
}

void assign_single(void *p) {
  struct flexible *__single s;
  s = p;
}

void assign_casted_single(void *p) {
  struct flexible *__single s;
  s = (struct flexible *)p;
}

void init_single_member_ref(void *p) {
  struct flexible *__single s = p; // expected-note{{'s' is initialized with a '__single' pointer}}
  s->count = 10; // expected-error{{assignment to 's->count' requires an immediately preceding assignment to 's' with a wide pointer}}
}

void init_casted_single_member_ref(void *p) {
  struct flexible *__single s = (struct flexible *)p; // expected-note{{'s' is initialized with a '__single' pointer}}
  s->count = 10; // expected-error{{assignment to 's->count' requires an immediately preceding assignment to 's' with a wide pointer}}
}

void assign_single_member_ref(void *p) {
  struct flexible *__single s;
  s = p; // expected-note{{'s' is initialized with a '__single' pointer}}
  s->count = 2; // expected-error{{assignment to 's->count' requires an immediately preceding assignment to 's' with a wide pointer}}
}

void assign_casted_single_member_ref(void *p) {
  struct flexible *__single s;
  s = (struct flexible *)p; // expected-note{{'s' is initialized with a '__single' pointer}}
  s->count = 7; // expected-error{{assignment to 's->count' requires an immediately preceding assignment to 's' with a wide pointer}}
}

void assign_casted_single_member_ref_bidi(void *p) {
  struct flexible *s;
  s = (struct flexible *)p;
  s->count = 7; // XXX: currently, flexible array member operations are analyzed only when their base is a single pointer type
}
