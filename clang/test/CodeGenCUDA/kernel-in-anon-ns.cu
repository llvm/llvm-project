// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -fcuda-is-device -cuid=abc \
// RUN:   -aux-triple x86_64-unknown-linux-gnu -std=c++11 -fgpu-rdc \
// RUN:   -emit-llvm -o - -x hip %s > %t.dev

// RUN: %clang_cc1 -triple x86_64-gnu-linux -cuid=abc \
// RUN:   -aux-triple amdgcn-amd-amdhsa -std=c++11 -fgpu-rdc \
// RUN:   -emit-llvm -o - -x hip %s > %t.host

// RUN: cat %t.dev %t.host | FileCheck -check-prefixes=HIP,COMMON %s

// RUN: echo "GPU binary" > %t.fatbin

// RUN: %clang_cc1 -triple nvptx -fcuda-is-device -cuid=abc \
// RUN:   -aux-triple x86_64-unknown-linux-gnu -std=c++11 -fgpu-rdc \
// RUN:   -emit-llvm -o - %s > %t.dev

// RUN: %clang_cc1 -triple x86_64-gnu-linux -cuid=abc \
// RUN:   -aux-triple nvptx -std=c++11 -fgpu-rdc -fcuda-include-gpubinary %t.fatbin \
// RUN:   -emit-llvm -o - %s > %t.host

// RUN: cat %t.dev %t.host | FileCheck -check-prefixes=CUDA,COMMON %s

#include "Inputs/cuda.h"

// HIP-DAG: define weak_odr {{.*}}void @[[KERN1:_ZN12_GLOBAL__N_16kernelEv\.intern\.b04fd23c98500190]](
// HIP-DAG: define weak_odr {{.*}}void @[[KERN2:_Z8tempKernIN12_GLOBAL__N_11XEEvT_\.intern\.b04fd23c98500190]](
// HIP-DAG: define weak_odr {{.*}}void @[[KERN3:_Z8tempKernIN12_GLOBAL__N_1UlvE_EEvT_\.intern\.b04fd23c98500190]](

// CUDA-DAG: define weak_odr {{.*}}void @[[KERN1:_ZN12_GLOBAL__N_16kernelEv__intern__b04fd23c98500190]](
// CUDA-DAG: define weak_odr {{.*}}void @[[KERN2:_Z8tempKernIN12_GLOBAL__N_11XEEvT___intern__b04fd23c98500190]](
// CUDA-DAG: define weak_odr {{.*}}void @[[KERN3:_Z8tempKernIN12_GLOBAL__N_1UlvE_EEvT___intern__b04fd23c98500190]](

// COMMON-DAG: @[[STR1:.*]] = {{.*}} c"[[KERN1]]\00"
// COMMON-DAG: @[[STR2:.*]] = {{.*}} c"[[KERN2]]\00"
// COMMON-DAG: @[[STR3:.*]] = {{.*}} c"[[KERN3]]\00"

// COMMON-DAG: call i32 @__{{.*}}RegisterFunction({{.*}}@[[STR1]]
// COMMON-DAG: call i32 @__{{.*}}RegisterFunction({{.*}}@[[STR2]]
// COMMON-DAG: call i32 @__{{.*}}RegisterFunction({{.*}}@[[STR3]]


template <typename T>
__global__ void tempKern(T x) {}

namespace {
  __global__ void kernel() {}
  struct X {};
  X x;
  auto lambda = [](){};
}

void test() {
  kernel<<<1, 1>>>();

  tempKern<<<1, 1>>>(x);

  tempKern<<<1, 1>>>(lambda);
}
