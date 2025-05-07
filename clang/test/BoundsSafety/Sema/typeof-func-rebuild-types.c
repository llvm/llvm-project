// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -verify %s
// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc -verify %s

#include <ptrcheck.h>

// Check if clang rebuilds param/return types of function declared with
// typeof/typedef.
// If the types are not rebuilt, the count expr in `bar` and `baz` will refer
// to `len` in `foo`, and the analysis will fail to catch the following errors.

void foo(int *__counted_by(len), int len);

__typeof__(foo) bar;

typedef void(baz_t)(int *__counted_by(len), int len);

baz_t baz;

void test(void) {
  // expected-note@+1 3{{'array' declared here}}
  int array[42];
  // expected-error@+1{{passing array 'array' (which has 42 elements) to parameter of type 'int *__single __counted_by(len)' (aka 'int *__single') with count value of 100 always fails}}
  foo(array, 100);
  // expected-error@+1{{passing array 'array' (which has 42 elements) to parameter of type 'int *__single __counted_by(len)' (aka 'int *__single') with count value of 100 always fails}}
  bar(array, 100);
  // expected-error@+1{{passing array 'array' (which has 42 elements) to parameter of type 'int *__single __counted_by(len)' (aka 'int *__single') with count value of 100 always fails}}
  baz(array, 100);
}
