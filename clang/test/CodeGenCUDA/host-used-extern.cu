// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -fcuda-is-device -x hip %s \
// RUN:   -fgpu-rdc -std=c++11 -emit-llvm -o - -target-cpu gfx906 | FileCheck %s

// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -fcuda-is-device -x hip %s \
// RUN:   -fgpu-rdc -std=c++11 -emit-llvm -o - -target-cpu gfx906 \
// RUN:   | FileCheck -check-prefix=NEG %s

// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -fcuda-is-device -x hip %s \
// RUN:   -std=c++11 -emit-llvm -o - -target-cpu gfx906 \
// RUN:   | FileCheck -check-prefixes=NEG,NORDC %s

#include "Inputs/cuda.h"

// CHECK-LABEL: @gpu.used.external = appending {{.*}}global
// CHECK-DAG: @_Z7kernel1v
// CHECK-DAG: @_Z7kernel4v
// CHECK-DAG: @var1
// CHECK-LABEL: @llvm.compiler.used = {{.*}} @gpu.used.external

// NEG-NOT: @gpu.used.external = {{.*}} @_Z7kernel2v
// NEG-NOT: @gpu.used.external = {{.*}} @_Z7kernel3v
// NEG-NOT: @gpu.used.external = {{.*}} @var2
// NEG-NOT: @gpu.used.external = {{.*}} @var3
// NORDC-NOT: @gpu.used.external = {{.*}} @_Z7kernel1v
// NORDC-NOT: @gpu.used.external = {{.*}} @_Z7kernel4v
// NORDC-NOT: @gpu.used.external = {{.*}} @var1

__global__ void kernel1();

// kernel2 is not marked as used since it is a definition.
__global__ void kernel2() {}

// kernel3 is not marked as used since it is not called by host function.
__global__ void kernel3();

// kernel4 is marked as used even though it is not called.
__global__ void kernel4();

extern __device__ int var1;

__device__ int var2;

extern __device__ int var3;

void use(int *p);

void test() {
  kernel1<<<1, 1>>>();
  void *p = (void*)kernel4;
  use(&var1);
}
