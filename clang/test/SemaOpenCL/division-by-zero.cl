// RUN: %clang_cc1 -triple spir-unknown-unknown -cl-std=CL3.0 -verify %s
// RUN: %clang_cc1 -triple spir-unknown-unknown -cl-std=clc++2021 -verify %s

// OpenCL C v3.0 section 6.5.1: an integer division or remainder by zero
// results in an unspecified value instead of undefined behavior. Keep
// warning: a literal zero divisor is almost always a bug.

void test(int x) {
  (void)(x / 0); // expected-warning{{division by zero results in an unspecified value}}
  (void)(x % 0); // expected-warning{{remainder by zero results in an unspecified value}}
  x /= 0;        // expected-warning{{division by zero results in an unspecified value}}
}
