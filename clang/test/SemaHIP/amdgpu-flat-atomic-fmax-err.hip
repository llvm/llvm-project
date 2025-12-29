// RUN: %clang_cc1 -fsyntax-only -triple amdgcn -target-cpu gfx942 -verify %s -fcuda-is-device
// RUN: %clang_cc1 -fsyntax-only -triple x86_64 -aux-triple amdgcn -verify %s
// expected-no-diagnostics

#define __device__ __attribute__((device))

__device__ void test_flat_atomic_fmax_f64_valid(double *ptr, double val) {
  double result;
  result = __builtin_amdgcn_flat_atomic_fmax_f64(ptr, val);
}

__device__ void test_flat_atomic_fmax_f64_errors(double *ptr, double val,
                                                 float *ptr_f) {
  double result;
  result = __builtin_amdgcn_flat_atomic_fmax_f64(ptr, val, 0);
  result = __builtin_amdgcn_flat_atomic_fmax_f64(ptr_f, val);
}

