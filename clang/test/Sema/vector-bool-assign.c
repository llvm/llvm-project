// RUN: %clang_cc1 -triple x86_64 -fsyntax-only -verify %s

typedef _Bool bool;

typedef __attribute__((ext_vector_type(8))) int v8i;
typedef __attribute__((ext_vector_type(8))) bool v8b;
typedef __attribute__((ext_vector_type(4))) float v4f;
typedef __attribute__((ext_vector_type(4))) bool v4b;

void foo(v8b);

v8b integral(v8i v) {
  v8b m1 = __builtin_convertvector(v, __attribute__((ext_vector_type(8))) int);
  v8b m2 = __builtin_convertvector(v, __attribute__((ext_vector_type(8))) unsigned);
  v8b m3 = __builtin_convertvector(v, __attribute__((ext_vector_type(8))) long);
  v8b m4 = __builtin_convertvector(v, __attribute__((ext_vector_type(8))) unsigned long);
  v8b m5 = __builtin_convertvector(v, __attribute__((ext_vector_type(8))) char);
  v8b m6 = __builtin_convertvector(v, __attribute__((ext_vector_type(8))) unsigned char);
  foo(v);
  return v;
}

v4b non_integral(v4f vf) {
  return vf; // expected-error{{returning 'v4f' (vector of 4 'float' values) from a function with incompatible result type 'v4b' (vector of 4 'bool' values}}
}

v4b size_mismatch(v8i v) {
  return v; // expected-error{{returning 'v8i' (vector of 8 'int' values) from a function with incompatible result type 'v4b' (vector of 4 'bool' values)}}
}
