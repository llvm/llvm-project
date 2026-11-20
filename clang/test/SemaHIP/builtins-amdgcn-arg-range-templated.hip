// REQUIRES: amdgpu-registered-target
// RUN: %clang_cc1 -triple amdgpu9.50-amd-amdhsa -fsyntax-only -fcuda-is-device -verify %s

#define __device__ __attribute__((device))

typedef _Float16 half2 __attribute__((ext_vector_type(2)));

template <int Sel>
__device__ half2 templated_sel(half2 Old, int Src, float Scale) {
  // expected-error@+1 {{argument value 4 is outside the valid range [0, 3]}}
  return __builtin_amdgcn_cvt_scalef32_f16_fp8(Old, Src, Scale, Sel, false);
}

template <typename T>
__device__ half2 templated_sizeof(half2 Old, int Src, float Scale) {
  // expected-error@+1 {{argument value 4 is outside the valid range [0, 3]}}
  return __builtin_amdgcn_cvt_scalef32_f16_fp8(Old, Src, Scale, sizeof(T), false);
}

__device__ void test(half2 Old, int Src, float Scale, half2 *Out) {
  Out[0] = templated_sel<3>(Old, Src, Scale);
  // expected-note@+1 {{in instantiation of function template specialization 'templated_sel<4>' requested here}}
  Out[1] = templated_sel<4>(Old, Src, Scale);
  Out[2] = templated_sizeof<char>(Old, Src, Scale);
  // expected-note@+1 {{in instantiation of function template specialization 'templated_sizeof<int>' requested here}}
  Out[3] = templated_sizeof<int>(Old, Src, Scale);
}
