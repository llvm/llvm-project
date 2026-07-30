// RUN: %clang_cc1 -verify %s
// RUN: %clang_cc1 -verify -cl-std=CL2.0 %s
// RUN: %clang_cc1 -verify -cl-std=clc++1.0 %s
// RUN: %clang_cc1 -verify -x c -D NOCL %s

#ifndef NOCL
kernel void f(__attribute__((nosvm)) global int* a);
#if (__OPENCL_C_VERSION__ == 200)
// expected-warning@-2 {{'nosvm' attribute is deprecated and ignored in OpenCL C version 2.0}}
#elif (__OPENCL_CPP_VERSION__ == 100)
// expected-warning@-4 {{'nosvm' attribute is deprecated and ignored in C++ for OpenCL version 1.0}}
#else
// expected-error@-6 {{attribute 'nosvm' is supported in the OpenCL version 2.0 onwards}}
#endif

__attribute__((nosvm)) void g(void); // expected-warning {{'nosvm' attribute only applies to variables}}

#else
void f(__attribute__((nosvm)) int* a); // expected-warning {{'nosvm' attribute ignored}}
#endif
