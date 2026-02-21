// RUN: %clang_cc1 -fsyntax-only -triple amdgcn -target-cpu gfx942 -verify %s -fcuda-is-device
// RUN: %clang_cc1 -fsyntax-only -triple x86_64 -aux-triple amdgcn -verify %s

#define __device__ __attribute__((device))
#define __shared__ __attribute__((shared))

__device__ double global_double;

__device__ void test_global_atomic_fmin_f64_valid(double *ptr, double val) {
  double result;
  result = __builtin_amdgcn_global_atomic_fmin_f64(ptr, val);
  result = __builtin_amdgcn_global_atomic_fmin_f64(&global_double, val);
}

__device__ void test_global_atomic_fmin_f64_errors(double *ptr, double val,
                                                   __shared__ double *lds_ptr,
                                                   float *ptr_f) {
  double result;
  result = __builtin_amdgcn_global_atomic_fmin_f64(ptr, val, 0); // expected-error{{too many arguments to function call, expected 2, have 3}}
  result = __builtin_amdgcn_global_atomic_fmin_f64(lds_ptr, val);
  result = __builtin_amdgcn_global_atomic_fmin_f64(ptr_f, val); // expected-error{{cannot initialize a parameter of type}}
}
