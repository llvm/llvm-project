// REQUIRES: amdgpu-registered-target
// RUN: %clang_cc1 -triple amdgcn-unknown-unknown -target-cpu gfx908 -verify -S -o - %s
// RUN: %clang_cc1 -triple amdgcn-unknown-unknown -target-cpu gfx90a -verify -S -o - %s

typedef int    v2i   __attribute__((ext_vector_type(2)));
typedef float  v4f   __attribute__((ext_vector_type(4)));
typedef half   v4h   __attribute__((ext_vector_type(4)));
typedef short  v4s   __attribute__((ext_vector_type(4)));
typedef int    v4i   __attribute__((ext_vector_type(4)));
typedef half   v8h   __attribute__((ext_vector_type(8)));
typedef short  v8s   __attribute__((ext_vector_type(8)));
typedef float v16f __attribute__((ext_vector_type(16)));
typedef int v16i __attribute__((ext_vector_type(16)));

void test_amdgcn_smfmac(global v4f *out4f,
                        global v16f *out16f,
                        global v4i *out4i,
                        global v16i *out16i,
                        v4h a4h, v4s a4s,
                        v8h b8h, v8s b8s,
                        v2i b2i,
                        v4f c4f,
                        v16f c16f,
                        v4i c4i,
                        v16i c16i,
                        int index)
{  
  *out4f = __builtin_amdgcn_smfmac_f32_16x16x32_f16(a4h, b8h, c4f, index, 0, 0); // expected-error{{'__builtin_amdgcn_smfmac_f32_16x16x32_f16' needs target feature gfx940-insts}}
  *out4f = __builtin_amdgcn_smfmac_f32_16x16x32_bf16(a4s, b8s, c4f, index, 0, 0); // expected-error{{'__builtin_amdgcn_smfmac_f32_16x16x32_bf16' needs target feature gfx940-insts}}
  *out16f = __builtin_amdgcn_smfmac_f32_32x32x16_f16(a4h, b8h, c16f, index, 0, 0); // expected-error{{'__builtin_amdgcn_smfmac_f32_32x32x16_f16' needs target feature gfx940-insts}}
  *out16f = __builtin_amdgcn_smfmac_f32_32x32x16_bf16(a4s, b8s, c16f, index, 0, 0); // expected-error{{'__builtin_amdgcn_smfmac_f32_32x32x16_bf16' needs target feature gfx940-insts}}
  *out4i = __builtin_amdgcn_smfmac_i32_16x16x64_i8(b2i, c4i, c4i, index, 0, 0); // expected-error{{'__builtin_amdgcn_smfmac_i32_16x16x64_i8' needs target feature gfx940-insts}}
  *out16i = __builtin_amdgcn_smfmac_i32_32x32x32_i8(b2i, c4i, c16i, index, 0, 0); // expected-error{{'__builtin_amdgcn_smfmac_i32_32x32x32_i8' needs target feature gfx940-insts}}
}
