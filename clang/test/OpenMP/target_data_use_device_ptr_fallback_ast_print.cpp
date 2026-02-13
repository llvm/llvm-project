// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=61 -ast-print %s | FileCheck %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=61 -x c++ -std=c++11 -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=61 -std=c++11 -include-pch %t -verify %s -ast-print | FileCheck %s

// expected-no-diagnostics

#ifndef HEADER
#define HEADER

// CHECK-LABEL:void f1(int *p, int *q)
void f1(int *p, int *q) {

// CHECK: #pragma omp target data use_device_ptr(fb_preserve: p)
#pragma omp target data use_device_ptr(fb_preserve: p)
  {}

// CHECK: #pragma omp target data use_device_ptr(fb_nullify: p)
#pragma omp target data use_device_ptr(fb_nullify: p)
  {}

// Without any fallback modifier
// CHECK: #pragma omp target data use_device_ptr(p)
#pragma omp target data use_device_ptr(p)
  {}

// Multiple variables with fb_preserve
// CHECK: #pragma omp target data use_device_ptr(fb_preserve: p,q)
#pragma omp target data use_device_ptr(fb_preserve: p, q)
  {}

// Multiple variables with fb_nullify
// CHECK: #pragma omp target data use_device_ptr(fb_nullify: p,q)
#pragma omp target data use_device_ptr(fb_nullify: p, q)
  {}
}
#endif
