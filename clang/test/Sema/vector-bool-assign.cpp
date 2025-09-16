// RUN: %clang_cc1 -triple x86_64 -fsyntax-only -verify %s

using v8i = int [[clang::ext_vector_type(8)]];
using v8b = bool [[clang::ext_vector_type(8)]];
using v4f = float [[clang::ext_vector_type(4)]];
using v4b = bool [[clang::ext_vector_type(4)]];

void foo(v8b);

v8b integral(v8i v) {
  v8b m1 = __builtin_convertvector(v, int [[clang::ext_vector_type(8)]]);
  v8b m2 = __builtin_convertvector(v, unsigned [[clang::ext_vector_type(8)]]);
  v8b m3 = __builtin_convertvector(v, long [[clang::ext_vector_type(8)]]);
  v8b m4 = __builtin_convertvector(v, unsigned long [[clang::ext_vector_type(8)]]);
  v8b m5 = __builtin_convertvector(v, char [[clang::ext_vector_type(8)]]);
  v8b m6 = __builtin_convertvector(v, unsigned char [[clang::ext_vector_type(8)]]);
  foo(v);
  return v;
}

v4b non_integral(v4f vf) {
  return vf; // expected-error{{cannot initialize return object of type 'v4b' (vector of 4 'bool' values) with an lvalue of type 'v4f' (vector of 4 'float' values)}}
}

v4b size_mismatch(v8i v) {
  return v; // expected-error{{cannot initialize return object of type 'v4b' (vector of 4 'bool' values) with an lvalue of type 'v8i' (vector of 8 'int' values)}}
}
