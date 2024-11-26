// RUN: %clang_cc1 -O0 -cl-std=CL2.0 -triple amdgcn-amd-amdhsa -target-cpu gfx906 -emit-llvm \
// RUN:   -verify -o - %s
// RUN: %clang_cc1 -O0 -cl-std=CL2.0 -triple amdgcn-amd-amdhsa -target-cpu gfx90a -emit-llvm \
// RUN:   -verify -o - %s
// RUN: %clang_cc1 -O0 -cl-std=CL2.0 -triple amdgcn-amd-amdhsa -target-cpu gfx940 -emit-llvm \
// RUN:   -verify -o - %s
// RUN: %clang_cc1 -O0 -cl-std=CL2.0 -triple amdgcn-amd-amdhsa -target-cpu gfx1200 -emit-llvm \
// RUN:   -verify -o - %s


// REQUIRES: amdgpu-registered-target

typedef unsigned int uint;
typedef unsigned int uint2 __attribute__((ext_vector_type(2)));
typedef half __attribute__((ext_vector_type(2))) half2;
typedef short __attribute__((ext_vector_type(2))) short2;

void test(global uint* out, global uint2* out_v2u32, uint a, uint b, global half2* out_v2f16, global float* out_f32, float scale, global short2* out_v2i16, float src0, float src1) {
  *out = __builtin_amdgcn_prng_b32(a); // expected-error{{'__builtin_amdgcn_prng_b32' needs target feature prng-inst}}
  *out_v2u32 = __builtin_amdgcn_permlane16_swap(a, b, false, false); // expected-error{{'__builtin_amdgcn_permlane16_swap' needs target feature permlane16-swap}}
  *out_v2u32 = __builtin_amdgcn_permlane32_swap(a, b, false, false); // expected-error{{'__builtin_amdgcn_permlane32_swap' needs target feature permlane32-swap}}
  *out_v2f16 = __builtin_amdgcn_cvt_scalef32_f16_fp8(*out_v2f16, a, scale, 0, false); // expected-error{{'__builtin_amdgcn_cvt_scalef32_f16_fp8' needs target feature fp8-cvt-scale-insts}}
  *out_f32 = __builtin_amdgcn_cvt_scalef32_f32_fp8(a, scale, 0); // expected-error{{'__builtin_amdgcn_cvt_scalef32_f32_fp8' needs target feature fp8-cvt-scale-insts}}
  *out_v2f16 = __builtin_amdgcn_cvt_scalef32_f16_bf8(*out_v2f16, a, scale, 0, false); // expected-error{{'__builtin_amdgcn_cvt_scalef32_f16_bf8' needs target feature bf8-cvt-scale-insts}}
  *out_f32 = __builtin_amdgcn_cvt_scalef32_f32_bf8(a, scale, 0); // expected-error{{'__builtin_amdgcn_cvt_scalef32_f32_bf8' needs target feature bf8-cvt-scale-insts}}
  *out_v2i16 = __builtin_amdgcn_cvt_scalef32_pk_fp8_f32(*out_v2i16, src0, src1, scale, true); // expected-error{{'__builtin_amdgcn_cvt_scalef32_pk_fp8_f32' needs target feature fp8-cvt-scale-insts}}
  *out_v2i16 = __builtin_amdgcn_cvt_scalef32_pk_bf8_f32(*out_v2i16, src0, src1, scale, true); // expected-error{{'__builtin_amdgcn_cvt_scalef32_pk_bf8_f32' needs target feature bf8-cvt-scale-insts}}
}
