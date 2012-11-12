// RUN: %clang_cc1 -emit-pch -o %t %s
// RUN: %clang_cc1 -include-pch %t -fsyntax-only %s 

#ifndef HEADER
#define HEADER
// Header.

#include "../SemaCUDA/cuda.h"

void kcall(void (*kp)()) {
  kp<<<1, 1>>>();
}

__global__ void kern() {
}

#else
// Using the header.

void test() {
  kcall(kern);
  kern<<<1, 1>>>();
}

#endif
