// RUN: %clang_cc1 -O0 -cl-std=CL2.0 -triple amdgcn-amd-amdhsa -target-cpu gfx906 -emit-llvm \
// RUN:   -verify -o - %s
// RUN: %clang_cc1 -O0 -cl-std=CL2.0 -triple amdgcn-amd-amdhsa -target-cpu gfx90a -emit-llvm \
// RUN:   -verify -o - %s
// RUN: %clang_cc1 -O0 -cl-std=CL2.0 -triple amdgcn-amd-amdhsa -target-cpu gfx942 -emit-llvm \
// RUN:   -verify -o - %s
// RUN: %clang_cc1 -O0 -cl-std=CL2.0 -triple amdgcn-amd-amdhsa -target-cpu gfx1200 -emit-llvm \
// RUN:   -verify -o - %s


// REQUIRES: amdgpu-registered-target

typedef unsigned int uint;
typedef unsigned short ushort;
typedef unsigned int uint2 __attribute__((ext_vector_type(2)));
typedef half __attribute__((ext_vector_type(2))) half2;
typedef short __attribute__((ext_vector_type(2))) short2;
typedef float __attribute__((ext_vector_type(2))) float2;
typedef __bf16 __attribute__((ext_vector_type(2))) bfloat2;
typedef float __attribute__((ext_vector_type(32))) float32;
typedef unsigned int __attribute__((ext_vector_type(6))) uint6;
typedef half __attribute__((ext_vector_type(32))) half32;
typedef __bf16 __attribute__((ext_vector_type(32))) bfloat32;

void test(global uint* out, global uint2* out_v2u32, uint a, uint b, uint c, global half2* out_v2f16, global float* out_f32, float scale, unsigned seed, global short2* out_v2i16, float src0, float src1,
          float2 src0_v2f32, global float2* out_v2f32, half2 src0_v2f16, bfloat2 src0_v2bf16, global bfloat2* out_v2bf16, global float32* out_v32f32, uint6 src_v6i32,
          global half32 *out_v32f16, global bfloat32 *out_v32bf16) {
  *out = __builtin_amdgcn_prng_b32(a); // expected-error{{'__builtin_amdgcn_prng_b32' needs target feature prng-inst}}
  *out_v2u32 = __builtin_amdgcn_permlane16_swap(a, b, false, false); // expected-error{{'__builtin_amdgcn_permlane16_swap' needs target feature permlane16-swap}}
  *out_v2u32 = __builtin_amdgcn_permlane32_swap(a, b, false, false); // expected-error{{'__builtin_amdgcn_permlane32_swap' needs target feature permlane32-swap}}
  *out_v2f16 = __builtin_amdgcn_cvt_scalef32_f16_fp8(*out_v2f16, a, scale, 0, false); // expected-error{{'__builtin_amdgcn_cvt_scalef32_f16_fp8' needs target feature fp8-cvt-scale-insts}}
  *out_f32 = __builtin_amdgcn_cvt_scalef32_f32_fp8(a, scale, 0); // expected-error{{'__builtin_amdgcn_cvt_scalef32_f32_fp8' needs target feature fp8-cvt-scale-insts}}
  *out_v2f16 = __builtin_amdgcn_cvt_scalef32_f16_bf8(*out_v2f16, a, scale, 0, false); // expected-error{{'__builtin_amdgcn_cvt_scalef32_f16_bf8' needs target feature bf8-cvt-scale-insts}}
  *out_f32 = __builtin_amdgcn_cvt_scalef32_f32_bf8(a, scale, 0); // expected-error{{'__builtin_amdgcn_cvt_scalef32_f32_bf8' needs target feature bf8-cvt-scale-insts}}
  *out_v2i16 = __builtin_amdgcn_cvt_scalef32_pk_fp8_f32(*out_v2i16, src0, src1, scale, true); // expected-error{{'__builtin_amdgcn_cvt_scalef32_pk_fp8_f32' needs target feature fp8-cvt-scale-insts}}
  *out_v2i16 = __builtin_amdgcn_cvt_scalef32_pk_bf8_f32(*out_v2i16, src0, src1, scale, true); // expected-error{{'__builtin_amdgcn_cvt_scalef32_pk_bf8_f32' needs target feature bf8-cvt-scale-insts}}
  *out_v2f32 = __builtin_amdgcn_cvt_scalef32_pk_f32_fp8(a, scale, true); // expected-error{{'__builtin_amdgcn_cvt_scalef32_pk_f32_fp8' needs target feature fp8-cvt-scale-insts}}
  *out_v2f32 = __builtin_amdgcn_cvt_scalef32_pk_f32_bf8(a, scale, true); // expected-error{{'__builtin_amdgcn_cvt_scalef32_pk_f32_bf8' needs target feature bf8-cvt-scale-insts}}
  *out_v2i16 = __builtin_amdgcn_cvt_scalef32_pk_fp8_f16(*out_v2i16, src0_v2f16, scale, true); // expected-error{{'__builtin_amdgcn_cvt_scalef32_pk_fp8_f16' needs target feature fp8-cvt-scale-insts}}
  *out_v2i16 = __builtin_amdgcn_cvt_scalef32_pk_fp8_bf16(*out_v2i16, src0_v2bf16, scale, true); // expected-error{{'__builtin_amdgcn_cvt_scalef32_pk_fp8_bf16' needs target feature fp8-cvt-scale-insts}}
  *out_v2i16 = __builtin_amdgcn_cvt_scalef32_pk_bf8_f16(*out_v2i16, src0_v2f16, scale, true); // expected-error{{'__builtin_amdgcn_cvt_scalef32_pk_bf8_f16' needs target feature bf8-cvt-scale-insts}}
  *out_v2i16 = __builtin_amdgcn_cvt_scalef32_pk_bf8_bf16(*out_v2i16, src0_v2bf16, scale, true); // expected-error{{'__builtin_amdgcn_cvt_scalef32_pk_bf8_bf16' needs target feature bf8-cvt-scale-insts}}
  *out_v2f32 = __builtin_amdgcn_cvt_scalef32_pk_f32_fp4(a, scale, 3); // expected-error{{'__builtin_amdgcn_cvt_scalef32_pk_f32_fp4' needs target feature fp4-cvt-scale-insts}}
  *out = __builtin_amdgcn_cvt_scalef32_pk_fp4_f32(*out, src0, src1, scale, 3); // expected-error{{'__builtin_amdgcn_cvt_scalef32_pk_fp4_f32' needs target feature fp4-cvt-scale-insts}}
  *out_v2f16 = __builtin_amdgcn_cvt_scalef32_pk_f16_fp4(a, scale, 3); // expected-error{{'__builtin_amdgcn_cvt_scalef32_pk_f16_fp4' needs target feature fp4-cvt-scale-insts}}
  *out_v2bf16 = __builtin_amdgcn_cvt_scalef32_pk_bf16_fp4(a, scale, 3); // expected-error{{'__builtin_amdgcn_cvt_scalef32_pk_bf16_fp4' needs target feature fp4-cvt-scale-insts}}
  *out_v32f32 = __builtin_amdgcn_cvt_scalef32_pk32_f32_fp6(src_v6i32, scale); // expected-error{{'__builtin_amdgcn_cvt_scalef32_pk32_f32_fp6' needs target feature fp6bf6-cvt-scale-insts}}
  *out_v32f32 = __builtin_amdgcn_cvt_scalef32_pk32_f32_bf6(src_v6i32, scale); // expected-error{{'__builtin_amdgcn_cvt_scalef32_pk32_f32_bf6' needs target feature fp6bf6-cvt-scale-insts}}
  *out_v32f16 = __builtin_amdgcn_cvt_scalef32_pk32_f16_fp6(src_v6i32, scale); // expected-error{{'__builtin_amdgcn_cvt_scalef32_pk32_f16_fp6' needs target feature fp6bf6-cvt-scale-insts}}
  *out_v32f16 = __builtin_amdgcn_cvt_scalef32_pk32_f16_bf6(src_v6i32, scale); // expected-error{{'__builtin_amdgcn_cvt_scalef32_pk32_f16_bf6' needs target feature fp6bf6-cvt-scale-insts}}
  *out_v32bf16 = __builtin_amdgcn_cvt_scalef32_pk32_bf16_fp6(src_v6i32, scale); // expected-error{{'__builtin_amdgcn_cvt_scalef32_pk32_bf16_fp6' needs target feature fp6bf6-cvt-scale-insts}}
  *out_v32bf16 = __builtin_amdgcn_cvt_scalef32_pk32_bf16_bf6(src_v6i32, scale); // expected-error{{'__builtin_amdgcn_cvt_scalef32_pk32_bf16_bf6' needs target feature fp6bf6-cvt-scale-insts}}
  *out_v2f16 = __builtin_amdgcn_cvt_scalef32_pk_f16_fp8(a, scale, true); // expected-error{{'__builtin_amdgcn_cvt_scalef32_pk_f16_fp8' needs target feature fp8-cvt-scale-insts}}
  *out_v2f16 = __builtin_amdgcn_cvt_scalef32_pk_f16_bf8(a, scale, true); // expected-error{{'__builtin_amdgcn_cvt_scalef32_pk_f16_bf8' needs target feature bf8-cvt-scale-insts}}
  *out_v2bf16 = __builtin_amdgcn_cvt_scalef32_pk_bf16_fp8(a, scale, true); // expected-error{{'__builtin_amdgcn_cvt_scalef32_pk_bf16_fp8' needs target feature fp8-cvt-scale-insts}}
  *out_v2bf16 = __builtin_amdgcn_cvt_scalef32_pk_bf16_bf8(a, scale, true); // expected-error{{'__builtin_amdgcn_cvt_scalef32_pk_bf16_bf8' needs target feature bf8-cvt-scale-insts}}
  *out = __builtin_amdgcn_cvt_scalef32_pk_fp4_f16(*out, src0_v2f16, scale, 3); // expected-error{{'__builtin_amdgcn_cvt_scalef32_pk_fp4_f16' needs target feature fp4-cvt-scale-insts}}
  *out = __builtin_amdgcn_cvt_scalef32_pk_fp4_bf16(*out, src0_v2bf16, scale, 3); // expected-error{{'__builtin_amdgcn_cvt_scalef32_pk_fp4_bf16' needs target feature fp4-cvt-scale-insts}}
  *out = __builtin_amdgcn_cvt_scalef32_sr_pk_fp4_f16(*out, src0_v2f16, 0, scale, 0); // expected-error{{'__builtin_amdgcn_cvt_scalef32_sr_pk_fp4_f16' needs target feature fp4-cvt-scale-insts}}
  *out = __builtin_amdgcn_cvt_scalef32_sr_pk_fp4_bf16(*out, src0_v2bf16, 0, scale, 0); // expected-error{{'__builtin_amdgcn_cvt_scalef32_sr_pk_fp4_bf16' needs target feature fp4-cvt-scale-insts}}
  *out = __builtin_amdgcn_cvt_scalef32_sr_pk_fp4_f32(*out, src0_v2f32, 0, scale, 0); // expected-error{{'__builtin_amdgcn_cvt_scalef32_sr_pk_fp4_f32' needs target feature fp4-cvt-scale-insts}}
  *out = __builtin_amdgcn_bitop3_b32(a, b, c, 1); // expected-error {{'__builtin_amdgcn_bitop3_b32' needs target feature bitop3-insts}}
  *out = __builtin_amdgcn_bitop3_b16((ushort)a, (ushort)b, (ushort)c, 1); // expected-error {{'__builtin_amdgcn_bitop3_b16' needs target feature bitop3-insts}}
  *out_v2bf16 = __builtin_amdgcn_cvt_sr_bf16_f32(*out_v2bf16, src0, seed, 0); // expected-error{{'__builtin_amdgcn_cvt_sr_bf16_f32' needs target feature f32-to-f16bf16-cvt-sr-insts}}
  *out_v2f16 = __builtin_amdgcn_cvt_sr_f16_f32(*out_v2f16, src0, seed, 0); // expected-error{{'__builtin_amdgcn_cvt_sr_f16_f32' needs target feature f32-to-f16bf16-cvt-sr-insts}}
}
