// RUN: %clang_cc1 %s -verify -pedantic -fsyntax-only -cl-std=CL1.1
// RUN: %clang_cc1 %s -verify -pedantic -fsyntax-only -cl-std=CL2.0

typedef float float4 __attribute__((ext_vector_type(4)));

void test_ext_vector_accessors(float4 V) {
  V = V.wzyx;
  V = V.abgr; // expected-warning {{vector component name 'a' is an OpenCL version 2.2 feature}}
  V = V.xyzr; // expected-warning {{vector component name 'r' is an OpenCL version 2.2 feature}} \
              // expected-error {{illegal vector component name 'r'}}
}
