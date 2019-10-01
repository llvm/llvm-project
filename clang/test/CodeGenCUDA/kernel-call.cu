// RUN: %clang_cc1 -target-sdk-version=8.0 -emit-llvm %s -o - \
// RUN: | FileCheck %s --check-prefixes=CUDA-OLD,CHECK
// RUN: %clang_cc1 -target-sdk-version=9.2  -emit-llvm %s -o - \
// RUN: | FileCheck %s --check-prefixes=CUDA-NEW,CHECK
// RUN: %clang_cc1 -x hip -emit-llvm %s -o - \
// RUN: | FileCheck %s --check-prefixes=HIP-OLD,CHECK
// RUN: %clang_cc1 -fhip-new-launch-api -x hip -emit-llvm %s -o - \
// RUN: | FileCheck %s --check-prefixes=HIP-NEW,CHECK

#include "Inputs/cuda.h"

// CHECK-LABEL: define{{.*}}g1
// HIP-OLD: call{{.*}}hipSetupArgument
// HIP-OLD: call{{.*}}hipLaunchByPtr
// HIP-NEW: call{{.*}}__hipPopCallConfiguration
// HIP-NEW: call{{.*}}hipLaunchKernel
// CUDA-OLD: call{{.*}}cudaSetupArgument
// CUDA-OLD: call{{.*}}cudaLaunch
// CUDA-NEW: call{{.*}}__cudaPopCallConfiguration
// CUDA-NEW: call{{.*}}cudaLaunchKernel
__global__ void g1(int x) {}

// CHECK-LABEL: define{{.*}}main
int main(void) {
  // HIP-OLD: call{{.*}}hipConfigureCall
  // HIP-NEW: call{{.*}}__hipPushCallConfiguration
  // CUDA-OLD: call{{.*}}cudaConfigureCall
  // CUDA-NEW: call{{.*}}__cudaPushCallConfiguration
  // CHECK: icmp
  // CHECK: br
  // CHECK: call{{.*}}g1
  g1<<<1, 1>>>(42);
}
