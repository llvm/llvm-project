// Based on clang/test/CodeGenCUDA/kernel-args.cu
// TODO: Add LLVM checks when cuda calling convention is supported

// RUN: %clang_cc1 -x cuda -triple nvptx64-nvidia-cuda -fcuda-is-device \
// RUN:   -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR

#include "Inputs/cuda.h"

struct A {
  int a[32];
  float *p;
};

// CIR: cir.func {{.*}} @_Z6kernel1A(
__global__ void kernel(A x) {
}

class Kernel {
public:
  // CIR: cir.func {{.*}} @_ZN6Kernel12memberKernelE1A(
  static __global__ void memberKernel(A x){}
  template<typename T> static __global__ void templateMemberKernel(T x) {}
};


template <typename T>
__global__ void templateKernel(T x) {}

void launch(void*);

void test() {
  Kernel K;
  // CIR: cir.func {{.*}} @_Z14templateKernelI1AEvT_(
  launch((void*)templateKernel<A>);

  // CIR: cir.func {{.*}} @_ZN6Kernel20templateMemberKernelI1AEEvT_(
  launch((void*)Kernel::templateMemberKernel<A>);
}

