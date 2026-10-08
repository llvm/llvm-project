// REQUIRES: amdgpu-registered-target
// RUN: %clang_cc1 -fsyntax-only -x hip -fcuda-is-device -verify=device -triple amdgpu %s
// RUN: %clang_cc1 -fsyntax-only -x cuda -fcuda-is-device -verify=device -triple nvptx64-nvidia-cuda %s
// RUN: %clang_cc1 -fsyntax-only -x hip -verify=host -triple x86_64-unknown-linux-gnu %s
// RUN: %clang_cc1 -fsyntax-only -x hip -fcuda-is-device -Wno-zero-length-shared-array -verify=host -triple amdgpu %s

// host-no-diagnostics

#define __device__ __attribute__((device))
#define __host__ __attribute__((host))
#define __global__ __attribute__((global))
#define __shared__ __attribute__((shared))

typedef float ZEROARR[0];

float global_array[0];
__device__ float device_array[0];
__shared__ float shared_global_array[0]; // device-warning {{zero-length __shared__ array 'shared_global_array' has no storage; did you mean 'extern __shared__' with an unspecified size for dynamic shared memory?}}

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
  __shared__ Ty shared_arr[Size]; // device-warning {{zero-length __shared__ array 'shared_arr' has no storage}}
}

template <typename Ty>
__device__ void dependent_element() {
  __shared__ Ty shared_arr[0]; // device-warning {{zero-length __shared__ array 'shared_arr' has no storage}}
}

template <typename Ty>
__device__ void dependent_type() {
  __shared__ Ty shared_arr; // device-warning {{zero-length __shared__ array 'shared_arr' has no storage}}
}

__host__ __device__ void host_dev_fun() {
  float array[0];
}

__device__ void device_fun() {
  __shared__ float array[0]; // device-warning {{zero-length __shared__ array 'array' has no storage}}
  __shared__ ZEROARR typedef_array; // device-warning {{zero-length __shared__ array 'typedef_array' has no storage}}
  templated<int, 0>(); // device-note {{in instantiation of function template specialization 'templated<int, 0U>' requested here}}
  templated<int, 1>();
  dependent_element<int>();
  dependent_element<float>();
  dependent_type<float[1]>();
  dependent_type<float[0]>(); // device-note {{in instantiation of function template specialization 'dependent_type<float[0]>' requested here}}
}
