// RUN: %clang_cc1 -target-sdk-version=8.0 -emit-llvm %s -o - \
// RUN: | FileCheck %s --check-prefixes=CUDA-OLD,CHECK
// RUN: %clang_cc1 -target-sdk-version=9.2  -emit-llvm %s -o - \
// RUN: | FileCheck %s --check-prefixes=CUDA-NEW,CHECK
// RUN: %clang_cc1 -x hip -emit-llvm %s -o - \
// RUN: | FileCheck %s --check-prefixes=HIP,CHECK


#include "Inputs/cuda.h"

// CHECK-LABEL: define{{.*}}g1
// HIP: call{{.*}}hipSetupArgument
// HIP: call{{.*}}hipLaunchByPtr
// CUDA-OLD: call{{.*}}cudaSetupArgument
// CUDA-OLD: call{{.*}}cudaLaunch
// CUDA-NEW: call{{.*}}__cudaPopCallConfiguration
// CUDA-NEW: call{{.*}}cudaLaunchKernel
__global__ void g1(int x) {}

// CHECK-LABEL: define{{.*}}main
int main(void) {
  // HIP: call{{.*}}hipConfigureCall
  // CUDA-OLD: call{{.*}}cudaConfigureCall
  // CUDA-NEW: call{{.*}}__cudaPushCallConfiguration
  // CHECK: icmp
  // CHECK: br
  // CHECK: call{{.*}}g1
  g1<<<1, 1>>>(42);
}
