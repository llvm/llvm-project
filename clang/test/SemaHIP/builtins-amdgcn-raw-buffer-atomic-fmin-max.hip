// RUN: %clang_cc1 -fsyntax-only -triple amdgcn -target-cpu gfx90a -verify %s -fcuda-is-device
// RUN: %clang_cc1 -fsyntax-only -triple x86_64 -aux-triple amdgcn -verify %s

#define __device__ __attribute__((device))

__device__ void test_raw_ptr_atomics(__amdgpu_buffer_rsrc_t rsrc, float f32, double f64, int offset, int soffset) {
  f32 = __builtin_amdgcn_raw_ptr_buffer_atomic_fmin_f32(f32, rsrc, offset, soffset, 0);
  f64 = __builtin_amdgcn_raw_ptr_buffer_atomic_fmin_f64(f64, rsrc, offset, soffset, 0);
  f32 = __builtin_amdgcn_raw_ptr_buffer_atomic_fmax_f32(f32, rsrc, offset, soffset, 0);
  f64 = __builtin_amdgcn_raw_ptr_buffer_atomic_fmax_f64(f64, rsrc, offset, soffset, 0);
}

__device__ void test_raw_ptr_atomics_err(__amdgpu_buffer_rsrc_t rsrc, float f32, double f64, int offset, int soffset) {
  f32 = __builtin_amdgcn_raw_ptr_buffer_atomic_fmin_f32(f32, rsrc, offset, soffset, 0, 4); // expected-error{{too many arguments to function call}}
  f64 = __builtin_amdgcn_raw_ptr_buffer_atomic_fmin_f64(f64, rsrc, offset, soffset, 0, 4); // expected-error{{too many arguments to function call}}
  f32 = __builtin_amdgcn_raw_ptr_buffer_atomic_fmax_f32(f32, rsrc, offset, soffset, 0, 4); // expected-error{{too many arguments to function call}}
  f64 = __builtin_amdgcn_raw_ptr_buffer_atomic_fmax_f64(f64, rsrc, offset, soffset, 0, 4); // expected-error{{too many arguments to function call}}
}
