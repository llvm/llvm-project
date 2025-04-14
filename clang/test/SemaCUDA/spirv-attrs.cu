// expected-no-diagnostics

// RUN: %clang_cc1 -triple spirv64 -aux-triple x86_64-unknown-linux-gnu \
// RUN:   -fcuda-is-device -verify -fsyntax-only %s

#define __global__ __attribute__((global))

__attribute__((reqd_work_group_size(128, 1, 1)))
__global__ void reqd_work_group_size_128_1_1() {}

template <unsigned a, unsigned b, unsigned c>
__attribute__((reqd_work_group_size(a, b, c)))
__global__ void reqd_work_group_size_a_b_c() {}

template <>
__global__ void reqd_work_group_size_a_b_c<128,1,1>(void);

__attribute__((work_group_size_hint(2, 2, 2)))
__global__ void work_group_size_hint_2_2_2() {}

template <unsigned a, unsigned b, unsigned c>
__attribute__((work_group_size_hint(a, b, c)))
__global__ void work_group_size_hint_a_b_c() {}

template <>
__global__ void work_group_size_hint_a_b_c<128,1,1>(void);

__attribute__((vec_type_hint(int)))
__global__ void vec_type_hint_int() {}

__attribute__((intel_reqd_sub_group_size(64)))
__global__ void intel_reqd_sub_group_size_64() {}
