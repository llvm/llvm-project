// RUN: %clang_cc1 -emit-pch -o %t %s
// RUN: %clang_cc1 -include-pch %t -fsyntax-only %s 
// RUN: %clang_cc1 -emit-pch -fcuda-is-device -o %t-device %s
// RUN: %clang_cc1 -fcuda-is-device -include-pch %t-device -fsyntax-only %s

#ifndef HEADER
#define HEADER
// Header.

#include "Inputs/cuda.h"

void kcall(void (*kp)()) {
  kp<<<1, 1>>>();
}

__global__ void kern() {
}

// Make sure that target overloaded functions remain
// available as overloads after PCH deserialization.
__host__ int overloaded_func();
__device__ int overloaded_func();

#else
// Using the header.

void test() {
  kcall(kern);
  kern<<<1, 1>>>();
  overloaded_func();
}

__device__ void test () {
  overloaded_func();
}
#endif
