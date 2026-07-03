// REQUIRES: amdgpu-registered-target
// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -target-cpu gfx12-5-generic -verify -S -o - %s

typedef short v2s __attribute__((ext_vector_type(2)));
typedef unsigned int v4u __attribute__((ext_vector_type(4)));
typedef float v4f  __attribute__((ext_vector_type(4)));
typedef half  v4h  __attribute__((ext_vector_type(4)));
typedef float v8f  __attribute__((ext_vector_type(8)));
typedef half  v8h  __attribute__((ext_vector_type(8)));
typedef half  v16h __attribute__((ext_vector_type(16)));
typedef __bf16 v16bf16 __attribute__((ext_vector_type(16)));
typedef __bf16 v32bf16 __attribute__((ext_vector_type(32)));

void test_amdgcn_swmmac_gfx12_5_generic(global v8f *out8f, global v4f *out4f,
                                         global float *outf,
                                         global unsigned int *outu,
                                         global unsigned long *outul,
                                         global v4u *outv4u,
                                         global v2s *outv2s,
                                         v8h a8h, v16h b16h, v4h a4h,
                                         v8h b8h, v8f c8f, v4f c4f,
                                         v16bf16 a16bf16, v32bf16 b32bf16,
                                         v4u v4u0,
                                         float x, float y, float z,
                                         unsigned int u0, unsigned int u1,
                                         unsigned int u2, unsigned long l0,
                                         unsigned long l1, int index) {
  *outf = __builtin_amdgcn_cubeid(x, y, z); // expected-error{{'__builtin_amdgcn_cubeid' needs target feature cube-insts}}
  *outv2s = __builtin_amdgcn_cvt_pknorm_i16(x, y); // expected-error{{'__builtin_amdgcn_cvt_pknorm_i16' needs target feature cvt-pknorm-vop2-insts}}
  *outu = __builtin_amdgcn_lerp(u0, u1, u2); // expected-error{{'__builtin_amdgcn_lerp' needs target feature lerp-inst}}
  *outul = __builtin_amdgcn_qsad_pk_u16_u8(l0, u0, l1); // expected-error{{'__builtin_amdgcn_qsad_pk_u16_u8' needs target feature qsad-insts}}
  *outu = __builtin_amdgcn_sad_u8(u0, u1, u2); // expected-error{{'__builtin_amdgcn_sad_u8' needs target feature sad-insts}}
  *outu = __builtin_amdgcn_msad_u8(u0, u1, u2); // expected-error{{'__builtin_amdgcn_msad_u8' needs target feature msad-insts}}
  *outul = __builtin_amdgcn_mqsad_pk_u16_u8(l0, u0, l1); // expected-error{{'__builtin_amdgcn_mqsad_pk_u16_u8' needs target feature mqsad-pk-insts}}
  *outv4u = __builtin_amdgcn_mqsad_u32_u8(l0, u0, v4u0); // expected-error{{'__builtin_amdgcn_mqsad_u32_u8' needs target feature mqsad-insts}}
  *out8f = __builtin_amdgcn_swmmac_f32_16x16x32_f16_w32(a8h, b16h, c8f, index); // expected-error{{'__builtin_amdgcn_swmmac_f32_16x16x32_f16_w32' needs target feature swmmac-gfx1200-insts,wavefrontsize32}}
  *out4f = __builtin_amdgcn_swmmac_f32_16x16x32_f16_w64(a4h, b8h, c4f, index); // expected-error{{'__builtin_amdgcn_swmmac_f32_16x16x32_f16_w64' needs target feature swmmac-gfx1200-insts,wavefrontsize64}}
  *out8f = __builtin_amdgcn_swmmac_f32_16x16x64_bf16(false, a16bf16, false, b32bf16, c8f, index, false, true); // expected-error{{'__builtin_amdgcn_swmmac_f32_16x16x64_bf16' needs target feature swmmac-gfx1250-insts,wavefrontsize32}}
}
