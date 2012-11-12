// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -fsyntax-only -fno-lax-vector-conversions -verify %s
typedef unsigned int __attribute__((vector_size (16))) vUInt32;
typedef int __attribute__((vector_size (16))) vSInt32;

vSInt32 foo (vUInt32 a) {
  vSInt32 b = { 0, 0, 0, 0 };
  b += a; // expected-error{{can't convert between vector values}}
  return b;
}
