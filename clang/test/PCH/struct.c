// Test this without pch.
// RUN: %clang_cc1 -include %S/struct.h -fsyntax-only -verify %s

// Test with pch.
// RUN: %clang_cc1 -emit-pch -o %t %S/struct.h
// RUN: %clang_cc1 -include-pch %t -fsyntax-only -verify %s 

struct Point *p1;

float getX(struct Point *p1) {
  return p1->x;
}

void *get_fun_ptr(void) {
  return fun->is_ptr? fun->ptr : 0;
}

struct Fun2 {
  int very_fun;
};

int get_very_fun(void) {
  return fun2->very_fun;
}

int *int_ptr_fail = &fun->is_ptr; // expected-error{{address of bit-field requested}}

struct Nested nested = { 1, 2 };
