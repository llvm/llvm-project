// RUN: %clang_cc1 -emit-llvm %s -o - | FileCheck %s

#include "../SemaCUDA/cuda.h"

__global__ void g1(int x) {}

int main(void) {
  // CHECK: call{{.*}}cudaConfigureCall
  // CHECK: icmp
  // CHECK: br
  // CHECK: call{{.*}}g1
  g1<<<1, 1>>>(42);
}
