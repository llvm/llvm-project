// RUN: %clang_cc1 -fsyntax-only -triple amdgcn -target-cpu gfx90a -verify %s -fcuda-is-device

#define __global__ __attribute__((global))
#define __device__ __attribute__((device))
#define __shared__ __attribute__((shared))

__shared__ short global_arr1[2147483648]; // expected-error {{'short[2147483648]' is too large for the address space (maximum allowed size of 4'294'967'295 bytes)}}
[[clang::loader_uninitialized]] short [[clang::address_space(3)]] global_arr1[2147483648]; // expected-error {{'__attribute__((address_space(3))) short[2147483648]' is too large for the address space (maximum allowed size of 4'294'967'295 bytes)}}

__device__ void func() {
  __shared__ int arr[1073741824]; // expected-error {{'int[1073741824]' is too large for the address space (maximum allowed size of 4'294'967'295 bytes)}}
}

__global__ void kernel() {
  __shared__ char arr[4294967296]; // expected-error {{'char[4294967296]' is too large for the address space (maximum allowed size of 4'294'967'295 bytes)}}
}

// TODO: The implementation of the __shared__ attribute doesn't check the
//       instantiation of dependent variables.
