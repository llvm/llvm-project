// REQUIRES: amdgpu-registered-target

// RUN: %clang_cc1 -triple amdgcn-unknown-unknown -target-cpu gfx1200 -verify -S -emit-llvm -o - %s

typedef unsigned int uint;

#pragma OPENCL EXTENSION cl_khr_fp64:enable

typedef float  v2f   __attribute__((ext_vector_type(2)));
typedef float  v4f   __attribute__((ext_vector_type(4)));
typedef float  v16f  __attribute__((ext_vector_type(16)));
typedef float  v32f  __attribute__((ext_vector_type(32)));
typedef half   v4h   __attribute__((ext_vector_type(4)));
typedef half   v8h   __attribute__((ext_vector_type(8)));
typedef half   v16h  __attribute__((ext_vector_type(16)));
typedef half   v32h  __attribute__((ext_vector_type(32)));
typedef int    v2i   __attribute__((ext_vector_type(2)));
typedef int    v4i   __attribute__((ext_vector_type(4)));
typedef int    v16i  __attribute__((ext_vector_type(16)));
typedef int    v32i  __attribute__((ext_vector_type(32)));
typedef short  v2s   __attribute__((ext_vector_type(2)));
typedef short  v4s   __attribute__((ext_vector_type(4)));
typedef short  v8s   __attribute__((ext_vector_type(8)));
typedef short  v16s  __attribute__((ext_vector_type(16)));
typedef short  v32s  __attribute__((ext_vector_type(32)));
typedef double v4d   __attribute__((ext_vector_type(4)));

void builtin_test_unsupported(double a_double, float a_float,
                              int a_int, long  a_long,
                              v4d a_v4d,
                              v2s a_v2s, v4s a_v4s, v8s a_v8s,
                              v2i a_v2i, v4i a_v4i, v16i a_v16i, v32i a_v32i,
                              v2f a_v2f, v4f a_v4f, v16f a_v16f, v32f  a_v32f,
                              v4h a_v4h, v8h a_v8h,

                              uint a, uint b) {

  __builtin_amdgcn_ds_gws_init(a, b); // expected-error {{'__builtin_amdgcn_ds_gws_init' needs target feature gws}}
  __builtin_amdgcn_ds_gws_barrier(a, b); // expected-error {{'__builtin_amdgcn_ds_gws_barrier' needs target feature gws}}
  __builtin_amdgcn_ds_gws_sema_v(a); // expected-error {{'__builtin_amdgcn_ds_gws_sema_v' needs target feature gws}}
  __builtin_amdgcn_ds_gws_sema_br(a, b); // expected-error {{'__builtin_amdgcn_ds_gws_sema_br' needs target feature gws}}
  __builtin_amdgcn_ds_gws_sema_p(a); // expected-error {{'__builtin_amdgcn_ds_gws_sema_p' needs target feature gws}}

  a_v32f = __builtin_amdgcn_mfma_f32_32x32x1f32(a_float, a_float, a_v32f, 0, 0, 0); // expected-error {{'__builtin_amdgcn_mfma_f32_32x32x1f32' needs target feature mai-insts}}
  a_v16f = __builtin_amdgcn_mfma_f32_16x16x1f32(a_float, a_float, a_v16f, 0, 0, 0); // expected-error {{'__builtin_amdgcn_mfma_f32_16x16x1f32' needs target feature mai-insts}}
  a_v4f =  __builtin_amdgcn_mfma_f32_4x4x1f32(a_float, a_float, a_v4f, 0, 0, 0); // expected-error {{'__builtin_amdgcn_mfma_f32_4x4x1f32' needs target feature mai-insts}}
  a_v16f = __builtin_amdgcn_mfma_f32_32x32x2f32(a_float, a_float, a_v16f, 0, 0, 0); // expected-error {{'__builtin_amdgcn_mfma_f32_32x32x2f32' needs target feature mai-insts}}
  a_v4f =  __builtin_amdgcn_mfma_f32_16x16x4f32(a_float, a_float, a_v4f, 0, 0, 0); // expected-error {{'__builtin_amdgcn_mfma_f32_16x16x4f32' needs target feature mai-insts}}
  a_v32f = __builtin_amdgcn_mfma_f32_32x32x4f16(a_v4h, a_v4h, a_v32f, 0, 0, 0); // expected-error {{'__builtin_amdgcn_mfma_f32_32x32x4f16' needs target feature mai-insts}}
  a_v16f = __builtin_amdgcn_mfma_f32_16x16x4f16(a_v4h, a_v4h, a_v16f, 0, 0, 0); // expected-error {{'__builtin_amdgcn_mfma_f32_16x16x4f16' needs target feature mai-insts}}
  a_v4f = __builtin_amdgcn_mfma_f32_4x4x4f16(a_v4h, a_v4h, a_v4f, 0, 0, 0); // expected-error {{'__builtin_amdgcn_mfma_f32_4x4x4f16' needs target feature mai-insts}}
  a_v16f = __builtin_amdgcn_mfma_f32_32x32x8f16(a_v4h, a_v4h, a_v16f, 0, 0, 0); // expected-error {{'__builtin_amdgcn_mfma_f32_32x32x8f16' needs target feature mai-insts}}
  a_v4f = __builtin_amdgcn_mfma_f32_16x16x16f16(a_v4h, a_v4h, a_v4f, 0, 0, 0); // expected-error {{'__builtin_amdgcn_mfma_f32_16x16x16f16' needs target feature mai-insts}}
  a_v32i = __builtin_amdgcn_mfma_i32_32x32x4i8(a_int, a_int, a_v32i, 0, 0, 0); // expected-error {{'__builtin_amdgcn_mfma_i32_32x32x4i8' needs target feature mai-insts}}
  a_v16i = __builtin_amdgcn_mfma_i32_16x16x4i8(a_int, a_int, a_v16i, 0, 0, 0); // expected-error {{'__builtin_amdgcn_mfma_i32_16x16x4i8' needs target feature mai-insts}}
  a_v4i = __builtin_amdgcn_mfma_i32_4x4x4i8(a_int, a_int, a_v4i, 0, 0, 0); // expected-error {{'__builtin_amdgcn_mfma_i32_4x4x4i8' needs target feature mai-insts}}
  a_v16i = __builtin_amdgcn_mfma_i32_32x32x8i8(a_int, a_int, a_v16i, 0, 0, 0); // expected-error {{'__builtin_amdgcn_mfma_i32_32x32x8i8' needs target feature mai-insts}}
  a_v4i = __builtin_amdgcn_mfma_i32_16x16x16i8(a_int, a_int, a_v4i, 0, 0, 0); // expected-error {{'__builtin_amdgcn_mfma_i32_16x16x16i8' needs target feature mai-insts}}
  a_v32f = __builtin_amdgcn_mfma_f32_32x32x2bf16(a_v2s, a_v2s, a_v32f, 0, 0, 0); // expected-error {{'__builtin_amdgcn_mfma_f32_32x32x2bf16' needs target feature mai-insts}}
  a_v16f = __builtin_amdgcn_mfma_f32_16x16x2bf16(a_v2s, a_v2s, a_v16f, 0, 0, 0); // expected-error {{'__builtin_amdgcn_mfma_f32_16x16x2bf16' needs target feature mai-insts}}
  a_v4f = __builtin_amdgcn_mfma_f32_4x4x2bf16(a_v2s, a_v2s, a_v4f, 0, 0, 0); // expected-error {{'__builtin_amdgcn_mfma_f32_4x4x2bf16' needs target feature mai-insts}}
  a_v16f = __builtin_amdgcn_mfma_f32_32x32x4bf16(a_v2s, a_v2s, a_v16f, 0, 0, 0); // expected-error {{'__builtin_amdgcn_mfma_f32_32x32x4bf16' needs target feature mai-insts}}
  a_v4f = __builtin_amdgcn_mfma_f32_16x16x8bf16(a_v2s, a_v2s, a_v4f, 0, 0, 0); // expected-error {{'__builtin_amdgcn_mfma_f32_16x16x8bf16' needs target feature mai-insts}}
  a_v32f = __builtin_amdgcn_mfma_f32_32x32x4bf16_1k(a_v4s, a_v4s, a_v32f, 0, 0, 0); // expected-error {{'__builtin_amdgcn_mfma_f32_32x32x4bf16_1k' needs target feature mai-insts}}
  a_v16f = __builtin_amdgcn_mfma_f32_16x16x4bf16_1k(a_v4s, a_v4s, a_v16f, 0, 0, 0); // expected-error {{'__builtin_amdgcn_mfma_f32_16x16x4bf16_1k' needs target feature mai-insts}}
  a_v4f = __builtin_amdgcn_mfma_f32_4x4x4bf16_1k(a_v4s, a_v4s, a_v4f, 0, 0, 0); // expected-error {{'__builtin_amdgcn_mfma_f32_4x4x4bf16_1k' needs target feature mai-insts}}
  a_v16f = __builtin_amdgcn_mfma_f32_32x32x8bf16_1k(a_v4s, a_v4s, a_v16f, 0, 0, 0); // expected-error {{'__builtin_amdgcn_mfma_f32_32x32x8bf16_1k' needs target feature mai-insts}}
  a_v4f = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(a_v4s, a_v4s, a_v4f, 0, 0, 0); // expected-error {{'__builtin_amdgcn_mfma_f32_16x16x16bf16_1k' needs target feature mai-insts}}
  a_v4d = __builtin_amdgcn_mfma_f64_16x16x4f64(a_double, a_double, a_v4d, 0, 0, 0); // expected-error {{'__builtin_amdgcn_mfma_f64_16x16x4f64' needs target feature mai-insts}}
  a_double = __builtin_amdgcn_mfma_f64_4x4x4f64(a_double, a_double, a_double, 0, 0, 0); // expected-error {{'__builtin_amdgcn_mfma_f64_4x4x4f64' needs target feature mai-insts}}
  a_v4i = __builtin_amdgcn_mfma_i32_16x16x32_i8(a_long, a_long, a_v4i, 0, 0, 0); // expected-error {{'__builtin_amdgcn_mfma_i32_16x16x32_i8' needs target feature mai-insts}}
  a_v16i = __builtin_amdgcn_mfma_i32_32x32x16_i8(a_long, a_long, a_v16i, 0, 0, 0); // expected-error {{'__builtin_amdgcn_mfma_i32_32x32x16_i8' needs target feature mai-insts}}
  a_v4f = __builtin_amdgcn_mfma_f32_16x16x8_xf32(a_v2f, a_v2f, a_v4f, 0, 0, 0); // expected-error {{'__builtin_amdgcn_mfma_f32_16x16x8_xf32' needs target feature mai-insts}}
  a_v16f = __builtin_amdgcn_mfma_f32_32x32x4_xf32(a_v2f, a_v2f, a_v16f, 0, 0, 0); // expected-error {{'__builtin_amdgcn_mfma_f32_32x32x4_xf32' needs target feature mai-insts}}
  a_v4f = __builtin_amdgcn_mfma_f32_16x16x32_bf8_bf8(a_long, a_long, a_v4f, 0, 0, 0); // expected-error {{'__builtin_amdgcn_mfma_f32_16x16x32_bf8_bf8' needs target feature fp8-insts}}
  a_v4f = __builtin_amdgcn_mfma_f32_16x16x32_bf8_fp8(a_long, a_long, a_v4f, 0, 0, 0); // expected-error {{'__builtin_amdgcn_mfma_f32_16x16x32_bf8_fp8' needs target feature fp8-insts}}
  a_v4f = __builtin_amdgcn_mfma_f32_16x16x32_fp8_bf8(a_long, a_long, a_v4f, 0, 0, 0); // expected-error {{'__builtin_amdgcn_mfma_f32_16x16x32_fp8_bf8' needs target feature fp8-insts}}
  a_v4f = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(a_long, a_long, a_v4f, 0, 0, 0); // expected-error {{'__builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8' needs target feature fp8-insts}}
  a_v16f = __builtin_amdgcn_mfma_f32_32x32x16_bf8_bf8(a_long, a_long, a_v16f, 0, 0, 0); // expected-error {{'__builtin_amdgcn_mfma_f32_32x32x16_bf8_bf8' needs target feature fp8-insts}}
  a_v16f = __builtin_amdgcn_mfma_f32_32x32x16_bf8_fp8(a_long, a_long, a_v16f, 0, 0, 0); // expected-error {{'__builtin_amdgcn_mfma_f32_32x32x16_bf8_fp8' needs target feature fp8-insts}}
  a_v16f = __builtin_amdgcn_mfma_f32_32x32x16_fp8_bf8(a_long, a_long, a_v16f, 0, 0, 0); // expected-error {{'__builtin_amdgcn_mfma_f32_32x32x16_fp8_bf8' needs target feature fp8-insts}}
  a_v16f = __builtin_amdgcn_mfma_f32_32x32x16_fp8_fp8(a_long, a_long, a_v16f, 0, 0, 0); // expected-error {{'__builtin_amdgcn_mfma_f32_32x32x16_fp8_fp8' needs target feature fp8-insts}}
  a_v4f = __builtin_amdgcn_smfmac_f32_16x16x32_f16(a_v4h, a_v8h, a_v4f, a_int, 0, 0); // expected-error {{'__builtin_amdgcn_smfmac_f32_16x16x32_f16' needs target feature mai-insts}}
  a_v16f = __builtin_amdgcn_smfmac_f32_32x32x16_f16(a_v4h, a_v8h, a_v16f, a_int, 0, 0); // expected-error {{'__builtin_amdgcn_smfmac_f32_32x32x16_f16' needs target feature mai-insts}}
  a_v4f = __builtin_amdgcn_smfmac_f32_16x16x32_bf16(a_v4s, a_v8s, a_v4f, a_int, 0, 0); // expected-error {{'__builtin_amdgcn_smfmac_f32_16x16x32_bf16' needs target feature mai-insts}}
  a_v16f = __builtin_amdgcn_smfmac_f32_32x32x16_bf16(a_v4s, a_v8s, a_v16f, a_int, 0, 0); // expected-error {{'__builtin_amdgcn_smfmac_f32_32x32x16_bf16' needs target feature mai-insts}}
  a_v4i = __builtin_amdgcn_smfmac_i32_16x16x64_i8(a_v2i, a_v4i, a_v4i, a_int, 0, 0); // expected-error {{'__builtin_amdgcn_smfmac_i32_16x16x64_i8' needs target feature mai-insts}}
  a_v16i = __builtin_amdgcn_smfmac_i32_32x32x32_i8(a_v2i, a_v4i, a_v16i, a_int, 0, 0); // expected-error {{'__builtin_amdgcn_smfmac_i32_32x32x32_i8' needs target feature mai-insts}}
  a_v4f = __builtin_amdgcn_smfmac_f32_16x16x64_bf8_bf8(a_v2i, a_v4i, a_v4f, a_int, 0, 0); // expected-error {{'__builtin_amdgcn_smfmac_f32_16x16x64_bf8_bf8' needs target feature fp8-insts}}
  a_v4f = __builtin_amdgcn_smfmac_f32_16x16x64_bf8_fp8(a_v2i, a_v4i, a_v4f, a_int, 0, 0); // expected-error {{'__builtin_amdgcn_smfmac_f32_16x16x64_bf8_fp8' needs target feature fp8-insts}}
  a_v4f = __builtin_amdgcn_smfmac_f32_16x16x64_fp8_bf8(a_v2i, a_v4i, a_v4f, a_int, 0, 0); // expected-error {{'__builtin_amdgcn_smfmac_f32_16x16x64_fp8_bf8' needs target feature fp8-insts}}
  a_v4f = __builtin_amdgcn_smfmac_f32_16x16x64_fp8_fp8(a_v2i, a_v4i, a_v4f, a_int, 0, 0); // expected-error {{'__builtin_amdgcn_smfmac_f32_16x16x64_fp8_fp8' needs target feature fp8-insts}}
  a_v16f = __builtin_amdgcn_smfmac_f32_32x32x32_bf8_bf8(a_v2i, a_v4i, a_v16f, a_int, 0, 0); // expected-error {{'__builtin_amdgcn_smfmac_f32_32x32x32_bf8_bf8' needs target feature fp8-insts}}
  a_v16f = __builtin_amdgcn_smfmac_f32_32x32x32_bf8_fp8(a_v2i, a_v4i, a_v16f, a_int, 0, 0); // expected-error {{'__builtin_amdgcn_smfmac_f32_32x32x32_bf8_fp8' needs target feature fp8-insts}}
  a_v16f = __builtin_amdgcn_smfmac_f32_32x32x32_fp8_bf8(a_v2i, a_v4i, a_v16f, a_int, 0, 0); // expected-error {{'__builtin_amdgcn_smfmac_f32_32x32x32_fp8_bf8' needs target feature fp8-insts}}
  a_v16f = __builtin_amdgcn_smfmac_f32_32x32x32_fp8_fp8(a_v2i, a_v4i, a_v16f, a_int, 0, 0); // expected-error {{'__builtin_amdgcn_smfmac_f32_32x32x32_fp8_fp8' needs target feature fp8-insts}}
}
