// RUN: %clang_cc1 -O0 -cl-std=CL2.0 -triple amdgcn-amd-amdhsa -target-cpu gfx1200 -target-feature +wavefrontsize32 -emit-llvm \
// RUN:   -verify -S -o - %s

// REQUIRES: amdgpu-registered-target

typedef half   v4h   __attribute__((ext_vector_type(4)));
typedef short  v4s   __attribute__((ext_vector_type(4)));
typedef __bf16 v4bf16 __attribute__((ext_vector_type(4)));

void amdgcn_global_load_tr(global int* int_inptr, global v4s* v4s_inptr, global v4h* v4h_inptr, global v4bf16* v4bf16_inptr)
{
  int out_1 = __builtin_amdgcn_global_load_tr_b64_i32(int_inptr); // expected-error{{'__builtin_amdgcn_global_load_tr_b64_i32' needs target feature gfx12-insts,wavefrontsize64}}
  v4s out_2 = __builtin_amdgcn_global_load_tr_b128_v4i16(v4s_inptr); // expected-error{{'__builtin_amdgcn_global_load_tr_b128_v4i16' needs target feature gfx12-insts,wavefrontsize64}}
  v4h out_3 = __builtin_amdgcn_global_load_tr_b128_v4f16(v4h_inptr); // expected-error{{'__builtin_amdgcn_global_load_tr_b128_v4f16' needs target feature gfx12-insts,wavefrontsize64}}
  v4bf16 o4 = __builtin_amdgcn_global_load_tr_b128_v4bf16(v4bf16_inptr); // expected-error{{'__builtin_amdgcn_global_load_tr_b128_v4bf16' needs target feature gfx12-insts,wavefrontsize64}}
}
