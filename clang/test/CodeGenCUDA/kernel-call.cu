// RUN: %clang_cc1 -emit-llvm %s -o - | FileCheck %s --check-prefixes=CUDA,CHECK
// RUN: %clang_cc1 -x hip -emit-llvm %s -o - | FileCheck %s --check-prefixes=HIP,CHECK


#include "Inputs/cuda.h"

// CHECK-LABEL: define{{.*}} void @_Z2g1i
// HIP: call{{.*}}hipSetupArgument
// HIP: call{{.*}}hipLaunchByPtr
// CUDA: call{{.*}}cudaSetupArgument
// CUDA: call{{.*}}cudaLaunch
__global__ void g1(int x) {}

// CHECK-LABEL: define{{.*}} i32 @main
int main(void) {
  // HIP: call{{.*}}hipConfigureCall
  // CUDA: call{{.*}}cudaConfigureCall
  // CHECK: icmp
  // CHECK: br
  // CHECK: call{{.*}}g1
  g1<<<1, 1>>>(42);
}
