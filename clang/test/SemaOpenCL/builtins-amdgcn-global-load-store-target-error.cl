// RUN: %clang_cc1 -cl-std=CL2.0 -triple amdgpu6.02-unknown-unknown -S -verify -o - %s
// RUN: %clang_cc1 -cl-std=CL2.0 -triple amdgpu7.05-unknown-unknown -S -verify -o - %s
// RUN: %clang_cc1 -cl-std=CL2.0 -triple amdgpu8.10-unknown-unknown -S -verify -o - %s
// REQUIRES: amdgpu-registered-target

typedef __attribute__((__vector_size__(4 * sizeof(unsigned int)))) unsigned int v4u32;
typedef v4u32 __global *global_ptr_to_v4u32;
typedef unsigned short u16;
typedef unsigned int u32;
typedef unsigned long u64;
typedef u16 __global *global_ptr_to_u16;
typedef u32 __global *global_ptr_to_u32;
typedef u64 __global *global_ptr_to_u64;

void test_amdgcn_av_load_store_target(global_ptr_to_u16 ptr16, u16 data16,
                                      global_ptr_to_u32 ptr32, u32 data32,
                                      global_ptr_to_u64 ptr64, u64 data64,
                                      global_ptr_to_v4u32 ptr128, v4u32 data128) {
  __builtin_amdgcn_av_load_b16(ptr16, 0); // expected-error{{'__builtin_amdgcn_av_load_b16' needs target feature flat-global-insts}}
  __builtin_amdgcn_av_load_b32(ptr32, 0); // expected-error{{'__builtin_amdgcn_av_load_b32' needs target feature flat-global-insts}}
  __builtin_amdgcn_av_load_b64(ptr64, 0); // expected-error{{'__builtin_amdgcn_av_load_b64' needs target feature flat-global-insts}}
  __builtin_amdgcn_av_load_b128(ptr128, 0); // expected-error{{'__builtin_amdgcn_av_load_b128' needs target feature flat-global-insts}}
  __builtin_amdgcn_av_store_b16(ptr16, data16, 0); // expected-error{{'__builtin_amdgcn_av_store_b16' needs target feature flat-global-insts}}
  __builtin_amdgcn_av_store_b32(ptr32, data32, 0); // expected-error{{'__builtin_amdgcn_av_store_b32' needs target feature flat-global-insts}}
  __builtin_amdgcn_av_store_b64(ptr64, data64, 0); // expected-error{{'__builtin_amdgcn_av_store_b64' needs target feature flat-global-insts}}
  __builtin_amdgcn_av_store_b128(ptr128, data128, 0); // expected-error{{'__builtin_amdgcn_av_store_b128' needs target feature flat-global-insts}}
}
