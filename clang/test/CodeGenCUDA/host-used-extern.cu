// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -fcuda-is-device -x hip %s \
// RUN:   -fgpu-rdc -std=c++11 -emit-llvm -o - -target-cpu gfx906 | FileCheck %s

// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -fcuda-is-device -x hip %s \
// RUN:   -fgpu-rdc -std=c++11 -emit-llvm -o - -target-cpu gfx906 \
// RUN:   | FileCheck -check-prefix=NEG %s

// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -fcuda-is-device -x hip %s \
// RUN:   -std=c++11 -emit-llvm -o - -target-cpu gfx906 \
// RUN:   | FileCheck -check-prefixes=NEG,NORDC %s

// RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -x hip %s \
// RUN:   -fgpu-rdc -std=c++11 -emit-llvm -o - \
// RUN:   | FileCheck -check-prefix=HOST-NEG %s


#include "Inputs/cuda.h"

// CHECK-LABEL: @__clang_gpu_used_external = internal {{.*}}global
// CHECK-DAG: @_Z7kernel1v
// CHECK-DAG: @_Z7kernel4v
// CHECK-DAG: @var1
// CHECK-LABEL: @llvm.compiler.used = {{.*}} @__clang_gpu_used_external

// NEG-NOT: @__clang_gpu_used_external = {{.*}} @_Z7kernel2v
// NEG-NOT: @__clang_gpu_used_external = {{.*}} @_Z7kernel3v
// NEG-NOT: @__clang_gpu_used_external = {{.*}} @_Z7kernel5v
// NEG-NOT: @__clang_gpu_used_external = {{.*}} @var2
// NEG-NOT: @__clang_gpu_used_external = {{.*}} @var3
// NEG-NOT: @__clang_gpu_used_external = {{.*}} @ext_shvar
// NEG-NOT: @__clang_gpu_used_external = {{.*}} @shvar
// NORDC-NOT: @__clang_gpu_used_external = {{.*}} @_Z7kernel1v
// NORDC-NOT: @__clang_gpu_used_external = {{.*}} @_Z7kernel4v
// NORDC-NOT: @__clang_gpu_used_external = {{.*}} @var1
// HOST-NEG-NOT: call void @__hipRegisterVar({{.*}}, ptr @ext_shvar
// HOST-NEG-NOT: call void @__hipRegisterVar({{.*}}, ptr @shvar
__global__ void kernel1();

// kernel2 is not marked as used since it is a definition.
__global__ void kernel2() {}

// kernel3 is not marked as used since it is not called by host function.
__global__ void kernel3();

// kernel4 is marked as used even though it is not called.
__global__ void kernel4();

// kernel5 is not marked as used since it is called by host function
// with weak_odr linkage, which may be dropped by linker.
__global__ void kernel5();

extern __device__ int var1;

__device__ int var2;

extern __device__ int var3;

void use(int *p);

void test() {
  kernel1<<<1, 1>>>();
  void *p = (void*)kernel4;
  use(&var1);
}

__global__ void test_lambda_using_extern_shared() {
  extern __shared__ int ext_shvar[];
  __shared__ int shvar[10];
  auto lambda = [&]() {
    ext_shvar[0] = 1;
    shvar[0] = 2;
  };
  lambda();
}

template<class T>
void template_caller() {
  kernel5<<<1, 1>>>();
  var1 = 1;
}

template void template_caller<int>();
