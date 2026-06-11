// RUN: %clang_cc1 -triple amdgcn-unknown-unknown -target-cpu verde -S -verify -o - %s
// REQUIRES: amdgpu-registered-target

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

typedef int v4i32 __attribute__((ext_vector_type(4)));
typedef int v2i32 __attribute__((ext_vector_type(2)));
typedef int v3i32 __attribute__((ext_vector_type(3)));
typedef int v8i32 __attribute__((ext_vector_type(8)));
typedef int v16i32 __attribute__((ext_vector_type(16)));
typedef float v2f32 __attribute__((ext_vector_type(2)));
typedef float v3f32 __attribute__((ext_vector_type(3)));
typedef float v4f32 __attribute__((ext_vector_type(4)));
typedef float v8f32 __attribute__((ext_vector_type(8)));
typedef float v16f32 __attribute__((ext_vector_type(16)));
typedef char v2i8 __attribute__((ext_vector_type(2)));
typedef char v3i8 __attribute__((ext_vector_type(3)));
typedef char v4i8 __attribute__((ext_vector_type(4)));
typedef half v2f16 __attribute__((ext_vector_type(2)));
typedef half v3f16 __attribute__((ext_vector_type(3)));
typedef half v4f16 __attribute__((ext_vector_type(4)));

int test_amdgcn_s_buffer_load_i32_non_const_aux(v4i32 rsrc, int offset, int aux) {
  return __builtin_amdgcn_s_buffer_load_i32(rsrc, offset, aux); //expected-error{{argument to '__builtin_amdgcn_s_buffer_load_i32' must be a constant integer}}
}

v2i32 test_amdgcn_s_buffer_load_v2i32_non_const_aux(v4i32 rsrc, int offset, int aux) {
  return __builtin_amdgcn_s_buffer_load_v2i32(rsrc, offset, aux); //expected-error{{argument to '__builtin_amdgcn_s_buffer_load_v2i32' must be a constant integer}}
}

v3i32 test_amdgcn_s_buffer_load_v3i32_non_const_aux(v4i32 rsrc, int offset, int aux) {
  return __builtin_amdgcn_s_buffer_load_v3i32(rsrc, offset, aux); //expected-error{{argument to '__builtin_amdgcn_s_buffer_load_v3i32' must be a constant integer}}
}

v4i32 test_amdgcn_s_buffer_load_v4i32_non_const_aux(v4i32 rsrc, int offset, int aux) {
  return __builtin_amdgcn_s_buffer_load_v4i32(rsrc, offset, aux); //expected-error{{argument to '__builtin_amdgcn_s_buffer_load_v4i32' must be a constant integer}}
}

v8i32 test_amdgcn_s_buffer_load_v8i32_non_const_aux(v4i32 rsrc, int offset, int aux) {
  return __builtin_amdgcn_s_buffer_load_v8i32(rsrc, offset, aux); //expected-error{{argument to '__builtin_amdgcn_s_buffer_load_v8i32' must be a constant integer}}
}

v16i32 test_amdgcn_s_buffer_load_v16i32_non_const_aux(v4i32 rsrc, int offset, int aux) {
  return __builtin_amdgcn_s_buffer_load_v16i32(rsrc, offset, aux); //expected-error{{argument to '__builtin_amdgcn_s_buffer_load_v16i32' must be a constant integer}}
}

float test_amdgcn_s_buffer_load_f32_non_const_aux(v4i32 rsrc, int offset, int aux) {
  return __builtin_amdgcn_s_buffer_load_f32(rsrc, offset, aux); //expected-error{{argument to '__builtin_amdgcn_s_buffer_load_f32' must be a constant integer}}
}

v2f32 test_amdgcn_s_buffer_load_v2f32_non_const_aux(v4i32 rsrc, int offset, int aux) {
  return __builtin_amdgcn_s_buffer_load_v2f32(rsrc, offset, aux); //expected-error{{argument to '__builtin_amdgcn_s_buffer_load_v2f32' must be a constant integer}}
}

v3f32 test_amdgcn_s_buffer_load_v3f32_non_const_aux(v4i32 rsrc, int offset, int aux) {
  return __builtin_amdgcn_s_buffer_load_v3f32(rsrc, offset, aux); //expected-error{{argument to '__builtin_amdgcn_s_buffer_load_v3f32' must be a constant integer}}
}

v4f32 test_amdgcn_s_buffer_load_v4f32_non_const_aux(v4i32 rsrc, int offset, int aux) {
  return __builtin_amdgcn_s_buffer_load_v4f32(rsrc, offset, aux); //expected-error{{argument to '__builtin_amdgcn_s_buffer_load_v4f32' must be a constant integer}}
}

v8f32 test_amdgcn_s_buffer_load_v8f32_non_const_aux(v4i32 rsrc, int offset, int aux) {
  return __builtin_amdgcn_s_buffer_load_v8f32(rsrc, offset, aux); //expected-error{{argument to '__builtin_amdgcn_s_buffer_load_v8f32' must be a constant integer}}
}

v16f32 test_amdgcn_s_buffer_load_v16f32_non_const_aux(v4i32 rsrc, int offset, int aux) {
  return __builtin_amdgcn_s_buffer_load_v16f32(rsrc, offset, aux); //expected-error{{argument to '__builtin_amdgcn_s_buffer_load_v16f32' must be a constant integer}}
}

char test_amdgcn_s_buffer_load_i8_non_const_aux(v4i32 rsrc, int offset, int aux) {
  return __builtin_amdgcn_s_buffer_load_i8(rsrc, offset, aux); //expected-error{{argument to '__builtin_amdgcn_s_buffer_load_i8' must be a constant integer}}
}

unsigned char test_amdgcn_s_buffer_load_u8_non_const_aux(v4i32 rsrc, int offset, int aux) {
  return __builtin_amdgcn_s_buffer_load_u8(rsrc, offset, aux); //expected-error{{argument to '__builtin_amdgcn_s_buffer_load_u8' must be a constant integer}}
}

short test_amdgcn_s_buffer_load_i16_non_const_aux(v4i32 rsrc, int offset, int aux) {
  return __builtin_amdgcn_s_buffer_load_i16(rsrc, offset, aux); //expected-error{{argument to '__builtin_amdgcn_s_buffer_load_i16' must be a constant integer}}
}

unsigned short test_amdgcn_s_buffer_load_u16_non_const_aux(v4i32 rsrc, int offset, int aux) {
  return __builtin_amdgcn_s_buffer_load_u16(rsrc, offset, aux); //expected-error{{argument to '__builtin_amdgcn_s_buffer_load_u16' must be a constant integer}}
}

v2i8 test_amdgcn_s_buffer_load_v2i8_non_const_aux(v4i32 rsrc, int offset, int aux) {
  return __builtin_amdgcn_s_buffer_load_v2i8(rsrc, offset, aux); //expected-error{{argument to '__builtin_amdgcn_s_buffer_load_v2i8' must be a constant integer}}
}

v3i8 test_amdgcn_s_buffer_load_v3i8_non_const_aux(v4i32 rsrc, int offset, int aux) {
  return __builtin_amdgcn_s_buffer_load_v3i8(rsrc, offset, aux); //expected-error{{argument to '__builtin_amdgcn_s_buffer_load_v3i8' must be a constant integer}}
}

v4i8 test_amdgcn_s_buffer_load_v4i8_non_const_aux(v4i32 rsrc, int offset, int aux) {
  return __builtin_amdgcn_s_buffer_load_v4i8(rsrc, offset, aux); //expected-error{{argument to '__builtin_amdgcn_s_buffer_load_v4i8' must be a constant integer}}
}

half test_amdgcn_s_buffer_load_f16_non_const_aux(v4i32 rsrc, int offset, int aux) {
  return __builtin_amdgcn_s_buffer_load_f16(rsrc, offset, aux); //expected-error{{argument to '__builtin_amdgcn_s_buffer_load_f16' must be a constant integer}}
}

v2f16 test_amdgcn_s_buffer_load_v2f16_non_const_aux(v4i32 rsrc, int offset, int aux) {
  return __builtin_amdgcn_s_buffer_load_v2f16(rsrc, offset, aux); //expected-error{{argument to '__builtin_amdgcn_s_buffer_load_v2f16' must be a constant integer}}
}

v3f16 test_amdgcn_s_buffer_load_v3f16_non_const_aux(v4i32 rsrc, int offset, int aux) {
  return __builtin_amdgcn_s_buffer_load_v3f16(rsrc, offset, aux); //expected-error{{argument to '__builtin_amdgcn_s_buffer_load_v3f16' must be a constant integer}}
}

v4f16 test_amdgcn_s_buffer_load_v4f16_non_const_aux(v4i32 rsrc, int offset, int aux) {
  return __builtin_amdgcn_s_buffer_load_v4f16(rsrc, offset, aux); //expected-error{{argument to '__builtin_amdgcn_s_buffer_load_v4f16' must be a constant integer}}
}

int test_amdgcn_s_buffer_load_i32_too_few_args(v4i32 rsrc) {
  return __builtin_amdgcn_s_buffer_load_i32(rsrc, 0); //expected-error{{too few arguments to function call, expected 3, have 2}}
}

int test_amdgcn_s_buffer_load_i32_too_many_args(v4i32 rsrc) {
  return __builtin_amdgcn_s_buffer_load_i32(rsrc, 0, 0, 0); //expected-error{{too many arguments to function call, expected 3, have 4}}
}

int test_amdgcn_s_buffer_load_i32_wrong_rsrc_type(__amdgpu_buffer_rsrc_t rsrc) {
  return __builtin_amdgcn_s_buffer_load_i32(rsrc, 0, 0); //expected-error{{passing '__private __amdgpu_buffer_rsrc_t' to parameter of incompatible type '__attribute__((__vector_size__(4 * sizeof(int)))) int' (vector of 4 'int' values)}}
}
