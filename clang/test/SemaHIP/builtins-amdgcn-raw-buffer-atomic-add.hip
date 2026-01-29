// RUN: %clang_cc1 -fsyntax-only -triple amdgcn -target-cpu gfx90a -verify %s -fcuda-is-device
// RUN: %clang_cc1 -fsyntax-only -triple x86_64 -aux-triple amdgcn -verify %s

typedef _Float16 __attribute__((ext_vector_type(2))) float16x2_t;

#define __device__ __attribute__((device))

__device__ void test_raw_ptr_atomics(__amdgpu_buffer_rsrc_t rsrc, int i32, float f32, float16x2_t v2f16, int offset, int soffset) {
  i32 = __builtin_amdgcn_raw_ptr_buffer_atomic_add_i32(i32, rsrc, offset, soffset, 0);
  f32 = __builtin_amdgcn_raw_ptr_buffer_atomic_fadd_f32(f32, rsrc, offset, soffset, 0);
  v2f16 = __builtin_amdgcn_raw_ptr_buffer_atomic_fadd_v2f16(v2f16, rsrc, offset, soffset, 0);
}

__device__ void test_raw_ptr_atomics_err(__amdgpu_buffer_rsrc_t rsrc, int i32, float f32, float16x2_t v2f16, int offset, int soffset) {
  i32 = __builtin_amdgcn_raw_ptr_buffer_atomic_add_i32(i32, rsrc, offset, soffset, 0, 4); // expected-error{{too many arguments to function call}}
  f32 = __builtin_amdgcn_raw_ptr_buffer_atomic_fadd_f32(f32, rsrc, offset, soffset, 0, 4); // expected-error{{too many arguments to function call}}
  v2f16 = __builtin_amdgcn_raw_ptr_buffer_atomic_fadd_v2f16(v2f16, rsrc, offset, soffset, 0, 4); // expected-error{{too many arguments to function call}}
}

__device__ void test_raw_ptr_atomics_f16_retty(__amdgpu_buffer_rsrc_t rsrc, int i32, float f32, float16x2_t v2f16, int offset, int soffset) {
  v2f16 = __builtin_amdgcn_raw_ptr_buffer_atomic_fadd_v2f16(v2f16, rsrc, offset, soffset, 0);
}
