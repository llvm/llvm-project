// RUN: %clang_cc1 -triple amdgcn -fcuda-is-device -emit-llvm %s -o - | FileCheck %s
#include "Inputs/cuda.h"

// CHECK: define amdgpu_kernel void @_ZN1A6kernelEv
class A {
public:
  static __global__ void kernel(){}
};

// CHECK: define void @_Z10non_kernelv
__device__ void non_kernel(){}

// CHECK: define amdgpu_kernel void @_Z6kerneli
__global__ void kernel(int x) {
  non_kernel();
}

// CHECK: define amdgpu_kernel void @_Z15template_kernelI1AEvT_
template<class T>
__global__ void template_kernel(T x) {}

void launch(void *f);

int main() {
  launch((void*)A::kernel);
  launch((void*)kernel);
  launch((void*)template_kernel<A>);
  return 0;
}
