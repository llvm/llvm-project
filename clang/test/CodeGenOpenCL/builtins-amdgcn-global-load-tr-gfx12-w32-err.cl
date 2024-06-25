// RUN: %clang_cc1 -O0 -cl-std=CL2.0 -triple amdgcn-amd-amdhsa -target-cpu gfx1200 -target-feature +wavefrontsize32 \
// RUN:   -verify -S -o - %s

// REQUIRES: amdgpu-registered-target

typedef short  v4s   __attribute__((ext_vector_type(4)));

void amdgcn_global_load_tr(global int* int_inptr, global v4s* v4s_inptr)
{
  int out_1 = __builtin_amdgcn_global_load_tr_b64_i32(int_inptr); // expected-error{{'__builtin_amdgcn_global_load_tr_b64_i32' needs target feature gfx12-insts,wavefrontsize64}}
  v4s out_2 = __builtin_amdgcn_global_load_tr_b128_v4i16(v4s_inptr); // expected-error{{'__builtin_amdgcn_global_load_tr_b128_v4i16' needs target feature gfx12-insts,wavefrontsize64}}
}
