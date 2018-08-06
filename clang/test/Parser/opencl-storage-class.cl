// RUN: %clang_cc1 %s -verify -fsyntax-only -triple spir-unknown-unknown

void test_storage_class_specs()
{
  static int a;   // expected-error {{OpenCL C version 1.0 does not support the 'static' storage class specifier}}
  register int b; // expected-error {{OpenCL C version 1.0 does not support the 'register' storage class specifier}}
  extern int c;   // expected-error {{OpenCL C version 1.0 does not support the 'extern' storage class specifier}}
  auto int d;     // expected-error {{OpenCL C version 1.0 does not support the 'auto' storage class specifier}}

#pragma OPENCL EXTENSION cl_clang_storage_class_specifiers : enable
  static int e; // expected-error {{static local variable must reside in constant address space}}
  register int f;
  extern int g; // expected-error {{extern variable must reside in constant address space}}
  auto int h;
}
