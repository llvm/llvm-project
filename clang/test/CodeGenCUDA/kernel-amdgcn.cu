// RUN: %clang_cc1 -triple amdgcn -fcuda-is-device -emit-llvm -x hip %s -o - | FileCheck %s
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

// CHECK: define amdgpu_kernel void @_Z11EmptyKernelIvEvv
template <typename T>
__global__ void EmptyKernel(void) {}

struct Dummy {
  /// Type definition of the EmptyKernel kernel entry point
  typedef void (*EmptyKernelPtr)();
  EmptyKernelPtr Empty() { return EmptyKernel<void>; } 
};

// CHECK: define amdgpu_kernel void @_Z15template_kernelI1AEvT_{{.*}} #[[ATTR:[0-9][0-9]*]]
template<class T>
__global__ void template_kernel(T x) {}

void launch(void *f);

int main() {
  Dummy D;
  launch((void*)A::kernel);
  launch((void*)kernel);
  launch((void*)template_kernel<A>);
  launch((void*)D.Empty());
  return 0;
}
// CHECK: attributes #[[ATTR]] = {{.*}}"amdgpu-flat-work-group-size"="1,1024"
