// RUN: %clang_cc1 %s -verify -fsyntax-only -cl-std=CL1.2
// RUN: %clang_cc1 %s -verify -fsyntax-only -cl-std=CL3.0

struct S {
  int x;
  int y[0]; // expected-error {{zero-length arrays are not permitted in OpenCL}}
};

global int g_zero_length_array[0]; // expected-error {{zero-length arrays are not permitted in OpenCL}}

kernel void foo(void) {
  int a[0]; // expected-error {{zero-length arrays are not permitted in OpenCL}}
  local int b[0]; // expected-error {{zero-length arrays are not permitted in OpenCL}}
}
