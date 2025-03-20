// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -fcuda-is-device -x hip %s \
// RUN:   -fgpu-rdc -std=c++11 -emit-llvm -o - -target-cpu gfx906 | FileCheck %s

#include "Inputs/cuda.h"

// CHECK-LABEL: @__clang_gpu_used_external = internal {{.*}}global
// References to the kernels must be in order of appearance.
// CHECK-SAME: [ptr @_Z6kernelILi3EEvPi, ptr @_Z6kernelILi1EEvPi, ptr @_Z6kernelILi2EEvPi, ptr @_Z6kernelILi0EEvPi]

template <int N>
__global__ void kernel(int* out) { *out = N; }

void host(int n) {
    void * k;
    switch (n) {
        case 3: k = (void*)&kernel<3>; break;
        case 1: k = (void*)&kernel<1>; break;
        case 2: k = (void*)&kernel<2>; break;
        case 0: k = (void*)&kernel<0>; break;
    }
}
