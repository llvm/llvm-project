// REQUIRES: amdgpu-registered-target
// RUN: %clang_cc1 -std=c++20 -triple amdgpu9.4-amd-amdhsa -fsyntax-only -fcuda-is-device -verify %s

#define __device__ __attribute__((device))

template <int CPol>
__device__ void test_buffer_inv_template() {
  __builtin_amdgcn_buffer_inv(CPol); // #policy
}

template __device__ void test_buffer_inv_template<0>();
template __device__ void test_buffer_inv_template<1>();
template __device__ void test_buffer_inv_template<16>();
template __device__ void test_buffer_inv_template<17>();

template __device__ void test_buffer_inv_template<2>(); // #invalid_policy
// expected-error@#policy {{must be a combination of the sc0 (1) and sc1 (16) cache-policy bits}}
// expected-note@#invalid_policy {{in instantiation of function template specialization 'test_buffer_inv_template<2>' requested here}}

template <typename T>
__device__ void test_buffer_inv_sizeof() {
  __builtin_amdgcn_buffer_inv(sizeof(T)); // #sizeof_policy
}

template __device__ void test_buffer_inv_sizeof<char>();
template __device__ void test_buffer_inv_sizeof<char[16]>();

template __device__ void test_buffer_inv_sizeof<char[2]>(); // #invalid_sizeof_policy
// expected-error@#sizeof_policy {{must be a combination of the sc0 (1) and sc1 (16) cache-policy bits}}
// expected-note@#invalid_sizeof_policy {{in instantiation of function template specialization 'test_buffer_inv_sizeof<char[2]>' requested here}}
