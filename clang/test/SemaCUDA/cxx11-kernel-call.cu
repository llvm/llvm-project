// RUN: %clang_cc1 -std=c++11 -fsyntax-only -verify %s

#include "Inputs/cuda.h"

__global__ void k1() {}

template<int ...Dimensions> void k1Wrapper() {
  void (*f)() = [] { k1<<<Dimensions, Dimensions>>>(); }; // expected-error {{initializer contains unexpanded parameter pack 'Dimensions'}}
  void (*g[])() = { [] { k1<<<Dimensions, Dimensions>>>(); } ... }; // ok
}
