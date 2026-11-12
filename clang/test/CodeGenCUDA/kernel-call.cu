// RUN: %clang_cc1 -target-sdk-version=8.0 -emit-llvm %s -o - \
// RUN: | FileCheck %s --check-prefixes=CUDA-OLD,CHECK
// RUN: %clang_cc1 -target-sdk-version=9.2  -emit-llvm %s -o - \
// RUN: | FileCheck %s --check-prefixes=CUDA-NEW,CHECK
// RUN: %clang_cc1 -target-sdk-version=9.2  -emit-llvm %s -o - \
// RUN:   -fgpu-default-stream=per-thread -DCUDA_API_PER_THREAD_DEFAULT_STREAM \
// RUN: | FileCheck %s --check-prefixes=CUDA-PTH,CHECK
// RUN: %clang_cc1 -x hip -emit-llvm %s -o - \
// RUN: | FileCheck %s --check-prefixes=HIP-OLD,CHECK
// RUN: %clang_cc1 -fhip-new-launch-api -x hip -emit-llvm %s -o - \
// RUN: | FileCheck %s --check-prefixes=HIP-NEW,LEGACY,CHECK
// RUN: %clang_cc1 -fhip-new-launch-api -x hip -emit-llvm %s -o - \
// RUN:   -fgpu-default-stream=legacy \
// RUN:   | FileCheck %s --check-prefixes=HIP-NEW,LEGACY,CHECK
// RUN: %clang_cc1 -fhip-new-launch-api -x hip -emit-llvm %s -o - \
// RUN:   -fgpu-default-stream=per-thread -DHIP_API_PER_THREAD_DEFAULT_STREAM \
// RUN:   | FileCheck %s --check-prefixes=HIP-NEW,PTH,CHECK

#include "Inputs/cuda.h"

// CHECK-LABEL: define{{.*}}g1
// HIP-OLD: call{{.*}}hipSetupArgument
// HIP-OLD: call{{.*}}hipLaunchByPtr
// HIP-NEW: call{{.*}}__hipPopCallConfiguration
// LEGACY: call{{.*}}hipLaunchKernel
// PTH: call{{.*}}hipLaunchKernel_spt
// CUDA-OLD: call{{.*}}cudaSetupArgument
// CUDA-OLD: call{{.*}}cudaLaunch
// CUDA-NEW: call{{.*}}__cudaPopCallConfiguration
// CUDA-NEW: call{{.*}}cudaLaunchKernel
// CUDA-PTH: call{{.*}}cudaLaunchKernel_ptsz
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
