// RUN: %clang_cc1 %s -triple spir-unknown-unknown -verify -pedantic -fsyntax-only -cl-std=CL1.0
// RUN: %clang_cc1 %s -triple spir-unknown-unknown -verify -pedantic -fsyntax-only -cl-std=CL1.1
// RUN: %clang_cc1 %s -triple spir-unknown-unknown -verify -fsyntax-only -cl-std=CL1.1 -DNOPEDANTIC
// RUN: %clang_cc1 %s -triple spir-unknown-unknown -verify -pedantic -fsyntax-only -cl-std=CL1.2 -DFP64

// Test with a target not supporting fp64.
// RUN: %clang_cc1 %s -cl-std=CL1.0 -triple r600-unknown-unknown -target-cpu r600 -verify -pedantic -fsyntax-only -DNOFP64 -DNOFP16
// RUN: %clang_cc1 -cl-std=CL3.0 %s -triple r600-unknown-unknown -target-cpu r600 -verify -pedantic -fsyntax-only -DNOFP64 -DNOFP16
// RUN: %clang_cc1 -cl-std=clc++2021 %s -triple r600-unknown-unknown -target-cpu r600 -verify -pedantic -fsyntax-only -DNOFP64 -DNOFP16

// Test with some extensions enabled or disabled by cmd-line args
//
// Target does not support fp64 and fp16 - override it
// RUN: %clang_cc1 %s -cl-std=CL1.0 -triple r600-unknown-unknown -target-cpu r600 -verify -pedantic -fsyntax-only -cl-ext=+cl_khr_fp64,+cl_khr_fp16
//
// Disable or enable all extensions
// RUN: %clang_cc1 %s -cl-std=CL1.0 -triple spir-unknown-unknown -verify -pedantic -fsyntax-only -cl-ext=-all -DNOFP64 -DNOFP16
// RUN: %clang_cc1 %s -cl-std=CL1.0 -triple r600-unknown-unknown -target-cpu r600 -verify -pedantic -fsyntax-only -cl-ext=+all
// RUN: %clang_cc1 %s -cl-std=CL1.0 -triple r600-unknown-unknown -target-cpu r600 -verify -pedantic -fsyntax-only -cl-ext=+all,-cl_khr_fp64 -DNOFP64
// RUN: %clang_cc1 %s -cl-std=CL1.0 -triple r600-unknown-unknown -target-cpu r600 -verify -pedantic -fsyntax-only -cl-ext=-all,+cl_khr_fp64 -DNOFP16
// RUN: %clang_cc1 -cl-std=CL3.0 %s -triple spir-unknown-unknown -verify -pedantic -fsyntax-only -cl-ext=-all -DNOFP64 -DNOFP16
// RUN: %clang_cc1 -cl-std=CL3.0 %s -triple r600-unknown-unknown -target-cpu r600 -verify -pedantic -fsyntax-only -cl-ext=+all -DFP64
// RUN: %clang_cc1 -cl-std=CL3.0 %s -triple r600-unknown-unknown -target-cpu r600 -verify -pedantic -fsyntax-only -cl-ext=+all,-__opencl_c_fp64,-cl_khr_fp64 -DNOFP64
// RUN: %clang_cc1 -cl-std=clc++2021 %s -triple spir-unknown-unknown -verify -pedantic -fsyntax-only -cl-ext=-all -DNOFP64 -DNOFP16
// RUN: %clang_cc1 -cl-std=clc++2021 %s -triple r600-unknown-unknown -target-cpu r600 -verify -pedantic -fsyntax-only -cl-ext=+all -DFP64
// RUN: %clang_cc1 -cl-std=clc++2021 %s -triple r600-unknown-unknown -target-cpu r600 -verify -pedantic -fsyntax-only -cl-ext=+all,-__opencl_c_fp64,-cl_khr_fp64 -DNOFP64
//
// Concatenating
// RUN: %clang_cc1 %s -cl-std=CL1.0 -triple spir-unknown-unknown -verify -pedantic -fsyntax-only -cl-ext=-cl_khr_fp64 -cl-ext=+cl_khr_fp64
// RUN: %clang_cc1 %s -cl-std=CL1.0 -triple spir-unknown-unknown -verify -pedantic -fsyntax-only -cl-ext=-cl_khr_fp64,+cl_khr_fp64
// RUN: %clang_cc1 %s -cl-std=CL1.0 -triple spir-unknown-unknown -verify -pedantic -fsyntax-only -cl-ext=-all -cl-ext=+cl_khr_fp64 -cl-ext=+cl_khr_fp16 -cl-ext=-cl_khr_fp64 -DNOFP64
// RUN: %clang_cc1 %s -triple spir-unknown-unknown -verify -pedantic -fsyntax-only -cl-ext=-all -cl-ext=+cl_khr_fp64,-cl_khr_fp64,+cl_khr_fp16 -DNOFP64
// RUN: %clang_cc1 -cl-std=CL3.0 %s -triple spir-unknown-unknown -verify -pedantic -fsyntax-only -cl-ext=-__opencl_c_fp64,-cl_khr_fp64 -cl-ext=+__opencl_c_fp64,+cl_khr_fp64 -DFP64
// RUN: %clang_cc1 -cl-std=CL3.0 %s -triple spir-unknown-unknown -verify -pedantic -fsyntax-only -cl-ext=-__opencl_c_fp64,+__opencl_c_fp64,+cl_khr_fp64 -DFP64
// RUN: %clang_cc1 -cl-std=CL3.0 %s -triple spir-unknown-unknown -verify -pedantic -fsyntax-only -cl-ext=-__opencl_c_fp64,-cl_khr_fp64 -DNOFP64
// RUN: %clang_cc1 -cl-std=clc++2021 %s -triple spir-unknown-unknown -verify -pedantic -fsyntax-only -cl-ext=-__opencl_c_fp64,-cl_khr_fp64 -cl-ext=+__opencl_c_fp64,+cl_khr_fp64 -DFP64
// RUN: %clang_cc1 -cl-std=clc++2021 %s -triple spir-unknown-unknown -verify -pedantic -fsyntax-only -cl-ext=-__opencl_c_fp64,+__opencl_c_fp64,+cl_khr_fp64 -DFP64
// RUN: %clang_cc1 -cl-std=clc++2021 %s -triple spir-unknown-unknown -verify -pedantic -fsyntax-only -cl-ext=-__opencl_c_fp64,-cl_khr_fp64 -DNOFP64

// Test with -finclude-default-header, which includes opencl-c.h. opencl-c.h
// disables all extensions by default, but supported core extensions for a
// particular OpenCL version must be re-enabled (for example, cl_khr_fp64 is
// enabled by default with -cl-std=CL2.0).
//
// RUN: %clang_cc1 %s -triple amdgcn-unknown-unknown -verify -pedantic -fsyntax-only -cl-std=CL2.0 -finclude-default-header
// RUN: %clang_cc1 %s -triple spir-unknown-unknown -verify -pedantic -fsyntax-only -cl-std=clc++1.0

#ifdef _OPENCL_H_
// expected-no-diagnostics
#endif

#ifdef FP64
// expected-no-diagnostics
#endif

#if __OPENCL_CPP_VERSION__ == 100
// expected-no-diagnostics
#endif

#if (defined(__OPENCL_C_VERSION__) && __OPENCL_C_VERSION__ < 120)
void f1(double da) {
#ifdef NOFP64
// expected-error@-2 {{type 'double' requires cl_khr_fp64 support}}
#elif !defined(NOPEDANTIC)
// expected-warning@-4{{Clang permits use of type 'double' regardless pragma if 'cl_khr_fp64' is supported}}
#endif
  double d;
#ifdef NOFP64
// expected-error@-2 {{type 'double' requires cl_khr_fp64 support}}
#elif !defined(NOPEDANTIC)
// expected-warning@-4{{Clang permits use of type 'double' regardless pragma if 'cl_khr_fp64' is supported}}
#endif
  // FIXME: this diagnostic depends on the extension pragma in the earlier versions.
  // There is no indication that this behavior is expected.
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
#if ((defined(__OPENCL_C_VERSION__) && __OPENCL_C_VERSION__ >= 300) || __OPENCL_CPP_VERSION__ >= 202100)
// expected-error@-3{{use of type 'double' requires cl_khr_fp64 and __opencl_c_fp64 support}}
#else
// expected-error@-5{{use of type 'double' requires cl_khr_fp64 support}}
#endif
#endif

  typedef double double4 __attribute__((ext_vector_type(4)));
  double4 d4 = {0.0f, 2.0f, 3.0f, 1.0f};
#ifdef NOFP64
#if ((defined(__OPENCL_C_VERSION__) && __OPENCL_C_VERSION__ >= 300) || __OPENCL_CPP_VERSION__ >= 202100)
// expected-error@-4 {{use of type 'double' requires cl_khr_fp64 and __opencl_c_fp64 support}}
#else
// expected-error@-6 {{use of type 'double' requires cl_khr_fp64 support}}
#endif
#endif

  (void) 1.0;
#ifdef NOFP64
#if ((defined(__OPENCL_C_VERSION__) && __OPENCL_C_VERSION__ >= 300) || __OPENCL_CPP_VERSION__ >= 202100)
// expected-warning@-3{{double precision constant requires cl_khr_fp64 and __opencl_c_fp64, casting to single precision}}
#else
// expected-warning@-5{{double precision constant requires cl_khr_fp64, casting to single precision}}
#endif
#endif
}

#pragma OPENCL EXTENSION cl_khr_fp64 : disable
#ifdef NOFP64
// expected-warning@-2{{unsupported OpenCL extension 'cl_khr_fp64' - ignoring}}
#endif

#if (defined(__OPENCL_C_VERSION__) && __OPENCL_C_VERSION__ < 120)
void f3(void) {
  double d;
#ifdef NOFP64
// expected-error@-2 {{type 'double' requires cl_khr_fp64 support}}
#elif !defined(NOPEDANTIC)
// expected-warning@-4 {{Clang permits use of type 'double' regardless pragma if 'cl_khr_fp64' is supported}}
#endif
}
#endif
