// REQUIRES: amdgpu-registered-target
// RUN: not %clang_cc1 -triple amdgcn-amd-amdhsa -target-cpu gfx906 -debug-info-kind=standalone -x hip -fcuda-is-device -emit-codegen-only < %s 2>&1 | FileCheck -DFILE=%s %s

#define __global__ __attribute__((global))
#define __shared__ __attribute__((shared))

template<typename T>
__global__ void use_huge_lds() {
  volatile __shared__ T huge[120000];
  huge[0] = 2;
}

// CHECK: <stdin>:[[@LINE-5]]:17: error: local memory (480000) exceeds limit (65536) in 'void use_huge_lds<int>()'
template
__global__ void use_huge_lds<int>();

// CHECK: <stdin>:[[@LINE-9]]:17: error: local memory (960000) exceeds limit (65536) in 'void use_huge_lds<double>()'
template
__global__ void use_huge_lds<double>();
