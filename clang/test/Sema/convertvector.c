// RUN: %clang_cc1 -fsyntax-only -verify %s

typedef double vector4double __attribute__((__vector_size__(32)));
typedef float  vector8float  __attribute__((__vector_size__(32)));

vector8float foo1(vector4double x) {
  return __builtin_convertvector(x, vector8float);  // expected-error {{same number of elements}}
}

float foo2(vector4double x) {
  return __builtin_convertvector(x, float);  // expected-error {{second argument to __builtin_convertvector must be of vector type}}
}

vector8float foo3(double x) {
  return __builtin_convertvector(x, vector8float);  // expected-error {{must be a vector}}
}

