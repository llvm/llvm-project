// RUN: %clang_cc1 -fsyntax-only -verify %s

#include "cuda.h"

__global__ void g1(int x) {}
__global__ int g2(int x) { // expected-error {{must have void return type}}
  return 1;
}
