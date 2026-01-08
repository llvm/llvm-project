// RUN: %clang_cc1 -std=c++20 -fsyntax-only -verify %s

// expected-no-diagnostics

#include "Inputs/cuda.h"

__device__ consteval int f() { return 0; }
int main() { return f(); }
