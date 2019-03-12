// RUN: %clang_cc1 %s -triple spir-unknown-unknown -verify -pedantic -fsyntax-only
// RUN: %clang_cc1 %s -triple spir-unknown-unknown -verify -pedantic -fsyntax-only -cl-std=CL1.1
// RUN: %clang_cc1 %s -triple spir-unknown-unknown -verify -pedantic -fsyntax-only -cl-std=CL1.2 -DFP64

// Test with a target not supporting fp64.
// RUN: %clang_cc1 %s -triple r600-unknown-unknown -target-cpu r600 -verify -pedantic -fsyntax-only -DNOFP64 -DNOFP16

// Test with some extensions enabled or disabled by cmd-line args
//
// Target does not support fp64 and fp16 - override it
// RUN: %clang_cc1 %s -triple r600-unknown-unknown -target-cpu r600 -verify -pedantic -fsyntax-only -cl-ext=+cl_khr_fp64,+cl_khr_fp16
//
// Disable or enable all extensions
// RUN: %clang_cc1 %s -triple spir-unknown-unknown -verify -pedantic -fsyntax-only -cl-ext=-all -DNOFP64 -DNOFP16
// RUN: %clang_cc1 %s -triple r600-unknown-unknown -target-cpu r600 -verify -pedantic -fsyntax-only -cl-ext=+all
// RUN: %clang_cc1 %s -triple r600-unknown-unknown -target-cpu r600 -verify -pedantic -fsyntax-only -cl-ext=+all,-cl_khr_fp64 -DNOFP64
// RUN: %clang_cc1 %s -triple r600-unknown-unknown -target-cpu r600 -verify -pedantic -fsyntax-only -cl-ext=-all,+cl_khr_fp64 -DNOFP16
//
// Concatenating
// RUN: %clang_cc1 %s -triple spir-unknown-unknown -verify -pedantic -fsyntax-only -cl-ext=-cl_khr_fp64 -cl-ext=+cl_khr_fp64
// RUN: %clang_cc1 %s -triple spir-unknown-unknown -verify -pedantic -fsyntax-only -cl-ext=-cl_khr_fp64,+cl_khr_fp64
// RUN: %clang_cc1 %s -triple spir-unknown-unknown -verify -pedantic -fsyntax-only -cl-ext=-all -cl-ext=+cl_khr_fp64 -cl-ext=+cl_khr_fp16 -cl-ext=-cl_khr_fp64 -DNOFP64
// RUN: %clang_cc1 %s -triple spir-unknown-unknown -verify -pedantic -fsyntax-only -cl-ext=-all -cl-ext=+cl_khr_fp64,-cl_khr_fp64,+cl_khr_fp16 -DNOFP64

// Test with -finclude-default-header, which includes opencl-c.h. opencl-c.h
// disables all extensions by default, but supported core extensions for a
// particular OpenCL version must be re-enabled (for example, cl_khr_fp64 is
// enabled by default with -cl-std=CL2.0).
//
// RUN: %clang_cc1 %s -triple amdgcn-unknown-unknown -verify -pedantic -fsyntax-only -cl-std=CL2.0 -finclude-default-header
// RUN: %clang_cc1 %s -triple spir-unknown-unknown -verify -pedantic -fsyntax-only -cl-std=c++ -finclude-default-header

#ifdef _OPENCL_H_
// expected-no-diagnostics
#endif

#ifdef FP64
// expected-no-diagnostics
#endif

#ifdef __OPENCL_CPP_VERSION__
// expected-no-diagnostics
#endif

#if (defined(__OPENCL_C_VERSION__) && __OPENCL_C_VERSION__ < 120)
void f1(double da) { // expected-error {{type 'double' requires cl_khr_fp64 extension}}
  double d; // expected-error {{type 'double' requires cl_khr_fp64 extension}}
  (void) 1.0; // expected-warning {{double precision constant requires cl_khr_fp64}}
}
#endif

#ifndef _OPENCL_H_
int isnan(float x) {
    return __builtin_isnan(x);
}

int isfinite(float x) {
    return __builtin_isfinite(x);
}
#endif

#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#ifdef NOFP64
// expected-warning@-2{{unsupported OpenCL extension 'cl_khr_fp64' - ignoring}}
#endif

#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#ifdef NOFP16
// expected-warning@-2{{unsupported OpenCL extension 'cl_khr_fp16' - ignoring}}
#endif

void f2(void) {
  double d;
#ifdef NOFP64
// expected-error@-2{{use of type 'double' requires cl_khr_fp64 extension to be enabled}}
#endif

  typedef double double4 __attribute__((ext_vector_type(4)));
  double4 d4 = {0.0f, 2.0f, 3.0f, 1.0f};
#ifdef NOFP64
// expected-error@-3 {{use of type 'double' requires cl_khr_fp64 extension to be enabled}}
// expected-error@-3 {{use of type 'double4' (vector of 4 'double' values) requires cl_khr_fp64 extension to be enabled}}
#endif

  (void) 1.0;

#ifdef NOFP64
// expected-warning@-3{{double precision constant requires cl_khr_fp64, casting to single precision}}
#endif
}

#pragma OPENCL EXTENSION cl_khr_fp64 : disable
#ifdef NOFP64
// expected-warning@-2{{unsupported OpenCL extension 'cl_khr_fp64' - ignoring}}
#endif

#if (defined(__OPENCL_C_VERSION__) && __OPENCL_C_VERSION__ < 120)
void f3(void) {
  double d; // expected-error {{type 'double' requires cl_khr_fp64 extension}}
}
#endif
