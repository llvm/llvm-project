// RUN: %clang_analyze_cc1 -analyzer-checker=core,unix.Malloc -verify %s

#include "Inputs/system-header-simulator-for-malloc.h"

struct Obj {
  int field;
};

void use(void *ptr);

void test_direct_param_uaf() {
  int *p = (int *)malloc(sizeof(int));
  free(p);
  use(p); // expected-warning{{Use of memory after it is released}}
}

void test_struct_field_uaf() {
  struct Obj *o = (struct Obj *)malloc(sizeof(struct Obj));
  free(o);
  use(&o->field); // expected-warning{{Use of memory after it is released}}
}

void test_no_warning_const_int() {
  use((void *)0x1234); // no-warning
}

void test_no_warning_stack() {
  int x = 42;
  use(&x); // no-warning
}

void test_nested_alloc() {
  struct Obj *o = (struct Obj *)malloc(sizeof(struct Obj));
  use(o);   // no-warning
  free(o);
  use(o);   // expected-warning{{Use of memory after it is released}}
}

void test_nested_field() {
    struct Obj *o = (struct Obj *)malloc(sizeof(struct Obj));
    int *f = &o->field;
    free(o);
    use(f); // expected-warning{{Use of memory after it is released}}
}
