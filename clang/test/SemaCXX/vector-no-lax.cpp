// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -fsyntax-only -fno-lax-vector-conversions -verify %s
typedef unsigned int __attribute__((vector_size (16))) vUInt32;
typedef int __attribute__((vector_size (16))) vSInt32;

vSInt32 foo (vUInt32 a) {
  vSInt32 b = { 0, 0, 0, 0 };
  b += a; // expected-error{{cannot convert between vector type 'vUInt32' (vector of 4 'unsigned int' values) and vector type 'vSInt32' (vector of 4 'int' values) as implicit conversion would cause truncation}}
  return b;
}
