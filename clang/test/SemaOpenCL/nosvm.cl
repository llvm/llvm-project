// RUN: %clang_cc1 -verify %s
// RUN: %clang_cc1 -verify -cl-std=CL2.0 -D CL20 %s
// RUN: %clang_cc1 -verify -x c -D NOCL %s

#ifndef NOCL
kernel void f(__attribute__((nosvm)) global int* a);
#ifndef CL20
// expected-error@-2 {{'nosvm' attribute requires OpenCL version 2.0}}
#else
// expected-warning@-4 {{'nosvm' attribute is deprecated and ignored in OpenCL version 2.0}}
#endif

__attribute__((nosvm)) void g(); // expected-warning {{'nosvm' attribute only applies to variables}}

#else
void f(__attribute__((nosvm)) int* a); // expected-warning {{'nosvm' attribute ignored}}
#endif
