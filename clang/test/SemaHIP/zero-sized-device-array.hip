// REQUIRES: amdgpu-registered-target
// RUN: %clang_cc1 -fsyntax-only -x hip -fcuda-is-device -verify -triple amdgpu %s
// RUN: %clang_cc1 -fsyntax-only -x hip -verify -triple x86_64-unknown-linux-gnu %s

// expected-no-diagnostics

#define __device__ __attribute__((device))
#define __host__ __attribute__((host))
#define __global__ __attribute__((global))
#define __shared__ __attribute__((shared))

typedef float ZEROARR[0];

float global_array[0];
__device__ float device_array[0];

__global__ void global_fun() {
  extern __shared__ float externArray[];
  ZEROARR TypeDef;
  float array[0];
}

__host__ void host_fun() {
  float array[0];
}

template <typename Ty, unsigned Size>
__device__ void templated() {
  Ty arr[Size];
  __shared__ Ty shared_arr[Size];
}

__host__ __device__ void host_dev_fun() {
  float array[0];
}

__device__ void device_fun() {
  __shared__ float array[0];
  templated<int, 0>();
}
