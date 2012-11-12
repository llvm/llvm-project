// RUN: %clang_cc1 -emit-pch -o %t %s
// RUN: %clang_cc1 -include-pch %t -fsyntax-only %s 

#ifndef HEADER
#define HEADER
// Header.

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

#else
// Using the header.

void test(void) {
  double d;
}

#endif
