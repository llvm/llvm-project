// RUN: %clang_cc1 -triple amdgcn-unknown-unknown -target-cpu verde -S -verify -o - %s
// REQUIRES: amdgpu-registered-target

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

typedef float v4f32 __attribute__((ext_vector_type(4)));
typedef half v4f16 __attribute__((ext_vector_type(4)));

v4f32 test_struct_buffer_load_format_v4f32(__amdgpu_buffer_rsrc_t rsrc, int vindex, int offset, int soffset, int aux) {
  return __builtin_amdgcn_struct_buffer_load_format_v4f32(rsrc, vindex, offset, soffset, aux); //expected-error{{argument to '__builtin_amdgcn_struct_buffer_load_format_v4f32' must be a constant integer}}
}

v4f16 test_struct_buffer_load_format_v4f16(__amdgpu_buffer_rsrc_t rsrc, int vindex, int offset, int soffset, int aux) {
  return __builtin_amdgcn_struct_buffer_load_format_v4f16(rsrc, vindex, offset, soffset, aux); //expected-error{{argument to '__builtin_amdgcn_struct_buffer_load_format_v4f16' must be a constant integer}}
}

void test_struct_buffer_store_format_v4f32(v4f32 vdata, __amdgpu_buffer_rsrc_t rsrc, int vindex, int offset, int soffset, int aux) {
  __builtin_amdgcn_struct_buffer_store_format_v4f32(vdata, rsrc, vindex, offset, soffset, aux); //expected-error{{argument to '__builtin_amdgcn_struct_buffer_store_format_v4f32' must be a constant integer}}
}

void test_struct_buffer_store_format_v4f16(v4f16 vdata, __amdgpu_buffer_rsrc_t rsrc, int vindex, int offset, int soffset, int aux) {
  __builtin_amdgcn_struct_buffer_store_format_v4f16(vdata, rsrc, vindex, offset, soffset, aux); //expected-error{{argument to '__builtin_amdgcn_struct_buffer_store_format_v4f16' must be a constant integer}}
}
