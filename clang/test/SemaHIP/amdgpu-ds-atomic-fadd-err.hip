// RUN: %clang_cc1 -fsyntax-only -triple amdgcn -target-cpu gfx90a -verify=device,both %s -fcuda-is-device
// RUN: %clang_cc1 -fsyntax-only -triple x86_64 -aux-triple amdgcn -verify=host,both %s

#define __device__ __attribute__((device))
#define __shared__ __attribute__((shared))

__device__ void test_ds_atomic_fadd_f32_valid(__shared__ float *lds_ptr, float val) {
  float result;
  result = __builtin_amdgcn_ds_atomic_fadd_f32(lds_ptr, val);
}

__device__ void test_ds_atomic_fadd_f32_errors(__shared__ float *lds_ptr, float val, 
                                                __shared__ double *lds_ptr_d,
                                                float *global_ptr) {
  float result;
  result = __builtin_amdgcn_ds_atomic_fadd_f32(lds_ptr, val, 0); // both-error{{too many arguments to function call, expected 2, have 3}}
  result = __builtin_amdgcn_ds_atomic_fadd_f32(global_ptr, val);
  result = __builtin_amdgcn_ds_atomic_fadd_f32(lds_ptr_d, val); // device-error{{cannot initialize a parameter of type '__shared__ float *' with an rvalue of type '__shared__ double *'}} host-error{{cannot initialize a parameter of type '__attribute__((address_space(3))) float *' with an rvalue of type '__attribute__((address_space(3))) double *'}}
}

__device__ void test_ds_atomic_fadd_f64_valid(__shared__ double *lds_ptr, double val) {
  double result;
  result = __builtin_amdgcn_ds_atomic_fadd_f64(lds_ptr, val);
}

__device__ void test_ds_atomic_fadd_f64_errors(__shared__ double *lds_ptr, double val,
                                                __shared__ float *lds_ptr_f,
                                                double *global_ptr) {
  double result;
  result = __builtin_amdgcn_ds_atomic_fadd_f64(lds_ptr, val, 0); // both-error{{too many arguments to function call, expected 2, have 3}}
  result = __builtin_amdgcn_ds_atomic_fadd_f64(global_ptr, val);
  result = __builtin_amdgcn_ds_atomic_fadd_f64(lds_ptr_f, val); // device-error{{cannot initialize a parameter of type '__shared__ double *' with an rvalue of type '__shared__ float *'}} host-error{{cannot initialize a parameter of type '__attribute__((address_space(3))) double *' with an rvalue of type '__attribute__((address_space(3))) float *'}}
}
