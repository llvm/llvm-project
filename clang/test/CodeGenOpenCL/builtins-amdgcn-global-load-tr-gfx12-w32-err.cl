// RUN: %clang_cc1 -O0 -cl-std=CL2.0 -triple amdgcn-amd-amdhsa -target-cpu gfx1200 -target-feature +wavefrontsize32 -emit-llvm \
// RUN:   -verify -S -o - %s

// REQUIRES: amdgpu-registered-target

typedef half   v4h   __attribute__((ext_vector_type(4)));
typedef short  v4s   __attribute__((ext_vector_type(4)));

void amdgcn_global_load_tr(global int* int_inptr, global v4s* v4s_inptr, global v4h* v4h_inptr)
{
  int out_4 = __builtin_amdgcn_global_load_tr_i32(int_inptr); // expected-error{{'__builtin_amdgcn_global_load_tr_i32' needs target feature gfx12-insts,wavefrontsize64}}
  v4s out_5 = __builtin_amdgcn_global_load_tr_v4i16(v4s_inptr); // expected-error{{'__builtin_amdgcn_global_load_tr_v4i16' needs target feature gfx12-insts,wavefrontsize64}}
  v4h out_6 = __builtin_amdgcn_global_load_tr_v4f16(v4h_inptr); // expected-error{{'__builtin_amdgcn_global_load_tr_v4f16' needs target feature gfx12-insts,wavefrontsize64}}
}

