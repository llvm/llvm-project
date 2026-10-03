// RUN: %clang_cc1 -fsyntax-only -verify %s

typedef double vector4double __attribute__((__vector_size__(32)));
typedef float  vector8float  __attribute__((__vector_size__(32)));

vector8float foo1(vector4double x) {
  return __builtin_splatvector(x, vector8float);  // expected-error {{first argument to __builtin_splatvector must be a single value}}
}

float foo2(vector4double x) {
  return __builtin_splatvector(x, float);  // expected-error {{first argument to __builtin_splatvector must be a single value}}
}

float foo4(float x) {
  return __builtin_splatvector(x, float); // expected-error {{second argument to __builtin_splatvector must be of vector type}}
}
