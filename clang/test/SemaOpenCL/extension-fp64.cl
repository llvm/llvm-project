// RUN: %clang_cc1 %s -verify -pedantic -fsyntax-only

void f1(double da) { // expected-error {{type 'double' requires cl_khr_fp64 extension}}
  double d; // expected-error {{type 'double' requires cl_khr_fp64 extension}}
  (void) 1.0; // expected-warning {{double precision constant requires cl_khr_fp64}}
}

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

void f2(void) {
  double d;
  (void) 1.0;
}

#pragma OPENCL EXTENSION cl_khr_fp64 : disable

void f3(void) {
  double d; // expected-error {{type 'double' requires cl_khr_fp64 extension}}
}
