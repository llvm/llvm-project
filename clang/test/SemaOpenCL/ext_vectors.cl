// RUN: %clang_cc1 %s -verify -pedantic -fsyntax-only -cl-std=CL1.1
// RUN: %clang_cc1 %s -verify -pedantic -fsyntax-only -cl-std=CL2.0
// RUN: %clang_cc1 %s -verify -pedantic -fsyntax-only -cl-std=CL3.0
// RUN: %clang_cc1 %s -verify -pedantic -fsyntax-only -cl-std=clc++1.0

typedef float float4 __attribute__((ext_vector_type(4)));

void test_ext_vector_accessors(float4 V) {
  V = V.wzyx;

  V = V.abgr;
#if (__OPENCL_C_VERSION__ < 300)
  // expected-warning@-2 {{vector component name 'a' is a feature from OpenCL version 3.0 onwards}}
#endif

  V = V.xyzr;
  // expected-error@-1 {{illegal vector component name 'r'}}
#if (__OPENCL_C_VERSION__ < 300)
  // expected-warning@-3 {{vector component name 'r' is a feature from OpenCL version 3.0 onwards}}
#endif
}
