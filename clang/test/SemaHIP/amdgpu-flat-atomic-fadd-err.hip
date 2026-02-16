// RUN: %clang_cc1 -fsyntax-only -triple amdgcn -target-cpu gfx942 -verify %s -fcuda-is-device
// RUN: %clang_cc1 -fsyntax-only -triple x86_64 -aux-triple amdgcn -verify %s

#define __device__ __attribute__((device))
#define __shared__ __attribute__((shared))

__device__ void test_flat_atomic_fadd_f32_valid(float *ptr, float val) {
  float result;
  result = __builtin_amdgcn_flat_atomic_fadd_f32(ptr, val);
}

__device__ void test_flat_atomic_fadd_f32_errors(float *ptr, float val,
                                                double *ptr_d) {
  float result;
  result = __builtin_amdgcn_flat_atomic_fadd_f32(ptr, val, 0); // expected-error{{too many arguments to function call, expected 2, have 3}}
  result = __builtin_amdgcn_flat_atomic_fadd_f32(ptr_d, val); // expected-error{{cannot initialize a parameter of type}}
}

__device__ void test_flat_atomic_fadd_f64_valid(double *ptr, double val) {
  double result;
  result = __builtin_amdgcn_flat_atomic_fadd_f64(ptr, val);
}

__device__ void test_flat_atomic_fadd_f64_errors(double *ptr, double val,
                                                float *ptr_f) {
  double result;
  result = __builtin_amdgcn_flat_atomic_fadd_f64(ptr, val, 0); // expected-error{{too many arguments to function call, expected 2, have 3}}
  result = __builtin_amdgcn_flat_atomic_fadd_f64(ptr_f, val); // expected-error{{cannot initialize a parameter of type}}
}
