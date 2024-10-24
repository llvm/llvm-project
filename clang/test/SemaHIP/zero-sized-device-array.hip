// REQUIRES: amdgpu-registered-target
// RUN:  %clang_cc1 -fsyntax-only -x hip -fcuda-is-device -verify -triple amdgcn %s
#define __device__ __attribute__((device))
#define __host__ __attribute__((host))
#define __global__ __attribute__((global))
#define __shared__ __attribute__((shared))

__global__ void global_fun() {
    float array[0];  // expected-error {{zero-length arrays are not permitted in HIP device code}}
}

// should not throw error for host side code.
__host__ void host_fun() {
    float array[0];
}

__host__ __device__ void host_dev_fun()
{
    float array[0]; // expected-error {{zero-length arrays are not permitted in HIP device code}}
}

__device__ void device_fun()
{
    __shared__ float array[0]; // expected-error {{zero-length arrays are not permitted in HIP device code}}
}
