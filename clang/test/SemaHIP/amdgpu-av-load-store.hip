// REQUIRES: amdgpu-registered-target
// RUN: %clang_cc1 -fsyntax-only -triple amdgpu9.50 -verify %s -fcuda-is-device
// RUN: %clang_cc1 -fsyntax-only -triple x86_64 -aux-triple amdgpu -verify %s

// expected-no-diagnostics

#define __device__ __attribute__((device))
#define __global__ __attribute__((global))

typedef __attribute__((__vector_size__(4 * sizeof(unsigned int)))) unsigned int v4u32;
typedef unsigned short u16;
typedef unsigned int u32;
typedef unsigned long u64;

__device__ void test_av_load_store_device(v4u32 *ptr, v4u32 data) {
  v4u32 res = __builtin_amdgcn_av_load_b128(ptr, 0);
  __builtin_amdgcn_av_store_b128(ptr, data, 0);
}

__global__ void test_av_load_store_kernel(v4u32 *ptr, v4u32 data) {
  v4u32 res = __builtin_amdgcn_av_load_b128(ptr, 0);
  __builtin_amdgcn_av_store_b128(ptr, data, 0);
}

__device__ void test_av_load_store_b16_device(u16 *ptr, u16 data) {
  u16 res = __builtin_amdgcn_av_load_b16(ptr, 0);
  __builtin_amdgcn_av_store_b16(ptr, data, 0);
}

__global__ void test_av_load_store_b16_kernel(u16 *ptr, u16 data) {
  u16 res = __builtin_amdgcn_av_load_b16(ptr, 0);
  __builtin_amdgcn_av_store_b16(ptr, data, 0);
}

__device__ void test_av_load_store_b32_device(u32 *ptr, u32 data) {
  u32 res = __builtin_amdgcn_av_load_b32(ptr, 0);
  __builtin_amdgcn_av_store_b32(ptr, data, 0);
}

__global__ void test_av_load_store_b32_kernel(u32 *ptr, u32 data) {
  u32 res = __builtin_amdgcn_av_load_b32(ptr, 0);
  __builtin_amdgcn_av_store_b32(ptr, data, 0);
}

__device__ void test_av_load_store_b64_device(u64 *ptr, u64 data) {
  u64 res = __builtin_amdgcn_av_load_b64(ptr, 0);
  __builtin_amdgcn_av_store_b64(ptr, data, 0);
}

__global__ void test_av_load_store_b64_kernel(u64 *ptr, u64 data) {
  u64 res = __builtin_amdgcn_av_load_b64(ptr, 0);
  __builtin_amdgcn_av_store_b64(ptr, data, 0);
}
