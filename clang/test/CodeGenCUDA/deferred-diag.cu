// RUN: not %clang_cc1 -std=c++11 -triple x86_64-unknown-linux-gnu \
// RUN:   -emit-llvm -o - %s 2>&1 | FileCheck %s
// RUN: not %clang_cc1 -std=c++11 -triple x86_64-unknown-linux-gnu \
// RUN:   -fcuda-is-device -emit-llvm -o - %s 2>&1 \
// RUN:   | FileCheck %s

// Check no crash due to deferred diagnostics.

#include "Inputs/cuda.h"

// CHECK: error: invalid output constraint '=h' in asm
// CHECK-NOT: core dump
inline __host__ __device__ int foo() {
  short h;
  __asm__("dont care" : "=h"(h) : "f"(0.0), "d"(0.0), "h"(0), "r"(0), "l"(0));
  return 0;
}

void host_fun() {
  foo();
}

__global__ void kernel() {
  foo();
}
