
// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -verify %s
// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc -verify %s

#include <ptrcheck.h>

void test(int *__null_terminated p, int v) {
  p++; // ok
  ++p; // ok
  p--; // expected-error{{cannot decrement '__terminated_by' pointer 'p'}}
  --p; // expected-error{{cannot decrement '__terminated_by' pointer 'p'}}

  p += 0;     // ok
  p -= 0;     // ok
  p += 1;     // ok
  p -= 1;     // expected-error{{pointer arithmetic on '__terminated_by' pointer 'p' can only increase the value by one}}
  p += 2;     // expected-error{{pointer arithmetic on '__terminated_by' pointer 'p' can only increase the value by one}}
  p -= 2;     // expected-error{{pointer arithmetic on '__terminated_by' pointer 'p' can only increase the value by one}}
  p += 2 - 1; // ok
  p += 1 - 2; // expected-error{{pointer arithmetic on '__terminated_by' pointer 'p' can only increase the value by one}}
  p -= 1 - 2; // ok
  p -= 2 - 1; // expected-error{{pointer arithmetic on '__terminated_by' pointer 'p' can only increase the value by one}}
  p += v;     // expected-error{{pointer arithmetic on '__terminated_by' pointer 'p' can only increase the value by one}}
  p -= v;     // expected-error{{pointer arithmetic on '__terminated_by' pointer 'p' can only increase the value by one}}

  (void)(p + 0);       // ok
  (void)(p - 0);       // ok
  (void)(p + 1);       // ok
  (void)(p - 1);       // expected-error{{pointer arithmetic on '__terminated_by' pointer 'p' can only increase the value by one}}
  (void)(p + 2);       // expected-error{{pointer arithmetic on '__terminated_by' pointer 'p' can only increase the value by one}}
  (void)(p - 2);       // expected-error{{pointer arithmetic on '__terminated_by' pointer 'p' can only increase the value by one}}
  (void)(p + (2 - 1)); // ok
  (void)(p + (1 - 2)); // expected-error{{pointer arithmetic on '__terminated_by' pointer 'p' can only increase the value by one}}
  (void)(p - (1 - 2)); // ok
  (void)(p - (2 - 1)); // expected-error{{pointer arithmetic on '__terminated_by' pointer 'p' can only increase the value by one}}
  (void)(p + v);       // expected-error{{pointer arithmetic on '__terminated_by' pointer 'p' can only increase the value by one}}
  (void)(p - v);       // expected-error{{pointer arithmetic on '__terminated_by' pointer 'p' can only increase the value by one}}

  (void)p[0];     // ok
  (void)p[1];     // expected-error{{array subscript on '__terminated_by' pointer 'p' is not allowed}}
  (void)p[-1];    // expected-error{{array subscript on '__terminated_by' pointer 'p' is not allowed}}
  (void)p[1 - 1]; // ok
  (void)p[2 - 1]; // expected-error{{array subscript on '__terminated_by' pointer 'p' is not allowed}}
  (void)p[1 - 2]; // expected-error{{array subscript on '__terminated_by' pointer 'p' is not allowed}}
  (void)p[v];     // expected-error{{array subscript on '__terminated_by' pointer 'p' is not allowed}}
}
