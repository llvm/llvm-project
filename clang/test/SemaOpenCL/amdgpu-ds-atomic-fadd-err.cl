// RUN: %clang_cc1 -fsyntax-only -cl-std=CL2.0 -triple amdgcn -target-cpu gfx90a -verify %s

kernel void test_ds_atomic_fadd_f32_valid(__local float *lds_ptr, float val) {
  float result;
  result = __builtin_amdgcn_ds_atomic_fadd_f32(lds_ptr, val);
}

kernel void test_ds_atomic_fadd_f32_errors(__local float *lds_ptr, float val, 
                                           __local double *lds_ptr_d,
                                           __global float *global_ptr) {
  float result;
  result = __builtin_amdgcn_ds_atomic_fadd_f32(lds_ptr, val, 0); // expected-error{{too many arguments to function call, expected 2, have 3}}
  result = __builtin_amdgcn_ds_atomic_fadd_f32(global_ptr, val); // expected-error{{passing '__global float *__private' to parameter of type '__local float *' changes address space of pointer}}
  result = __builtin_amdgcn_ds_atomic_fadd_f32(lds_ptr_d, val); // expected-error{{incompatible pointer types passing '__local double *__private' to parameter of type '__local float *'}}
}

kernel void test_ds_atomic_fadd_f64_valid(__local double *lds_ptr, double val) {
  double result;
  result = __builtin_amdgcn_ds_atomic_fadd_f64(lds_ptr, val);
}

kernel void test_ds_atomic_fadd_f64_errors(__local double *lds_ptr, double val,
                                           __local float *lds_ptr_f,
                                           __global double *global_ptr) {
  double result;
  result = __builtin_amdgcn_ds_atomic_fadd_f64(lds_ptr, val, 0); // expected-error{{too many arguments to function call, expected 2, have 3}}
  result = __builtin_amdgcn_ds_atomic_fadd_f64(global_ptr, val);// expected-error{{passing '__global double *__private' to parameter of type '__local double *' changes address space of pointer}}
  result = __builtin_amdgcn_ds_atomic_fadd_f64(lds_ptr_f, val); // expected-error{{incompatible pointer types passing '__local float *__private' to parameter of type '__local double *'}}
}
