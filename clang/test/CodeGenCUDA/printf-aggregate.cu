// REQUIRES: x86-registered-target
// REQUIRES: nvptx-registered-target

// RUN: not %clang_cc1 -triple nvptx64-nvidia-cuda -fcuda-is-device -emit-llvm \
// RUN:   -o - %s 2>&1 | FileCheck %s

#include "Inputs/cuda.h"

// Check that we don't crash when asked to printf a non-scalar arg.
struct Struct {
  int x;
  int y;
};
__device__ void PrintfNonScalar() {
  // CHECK: cannot compile this non-scalar arg to printf
  printf("%d", Struct());
}
