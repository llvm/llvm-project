// REQUIRES: amdgpu-registered-target
// RUN: %clang_cc1 -fsyntax-only -triple amdgcn -target-cpu gfx950 -verify %s -fcuda-is-device
// RUN: %clang_cc1 -fsyntax-only -triple x86_64 -aux-triple amdgcn -verify %s

// expected-no-diagnostics

#define __device__ __attribute__((device))
#define __global__ __attribute__((global))

typedef __attribute__((__vector_size__(4 * sizeof(unsigned int)))) unsigned int v4u32;

__device__ void test_av_load_store_device(v4u32 *ptr, v4u32 data) {
  v4u32 res = __builtin_amdgcn_av_load_b128(ptr, 0);
  __builtin_amdgcn_av_store_b128(ptr, data, 0);
}

__global__ void test_av_load_store_kernel(v4u32 *ptr, v4u32 data) {
  v4u32 res = __builtin_amdgcn_av_load_b128(ptr, 0);
  __builtin_amdgcn_av_store_b128(ptr, data, 0);
}
