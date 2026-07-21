// RUN: split-file %s %t
// RUN: %clang_cc1 -cl-std=CL2.0 -triple amdgcn-unknown-unknown -target-cpu gfx602 -S -verify -o - %t/load.cl
// RUN: %clang_cc1 -cl-std=CL2.0 -triple amdgcn-unknown-unknown -target-cpu gfx705 -S -verify -o - %t/load.cl
// RUN: %clang_cc1 -cl-std=CL2.0 -triple amdgcn-unknown-unknown -target-cpu gfx810 -S -verify -o - %t/load.cl
// RUN: %clang_cc1 -cl-std=CL2.0 -triple amdgcn-unknown-unknown -target-cpu gfx602 -S -verify -o - %t/store.cl
// RUN: %clang_cc1 -cl-std=CL2.0 -triple amdgcn-unknown-unknown -target-cpu gfx705 -S -verify -o - %t/store.cl
// RUN: %clang_cc1 -cl-std=CL2.0 -triple amdgcn-unknown-unknown -target-cpu gfx810 -S -verify -o - %t/store.cl
// REQUIRES: amdgpu-registered-target

//--- load.cl
typedef __attribute__((__vector_size__(4 * sizeof(unsigned int)))) unsigned int v4u32;
typedef v4u32 __global *global_ptr_to_v4u32;

v4u32 test_amdgcn_av_load_b128_target(global_ptr_to_v4u32 ptr) {
  return __builtin_amdgcn_av_load_b128(ptr, 0); // expected-error{{'__builtin_amdgcn_av_load_b128' needs target feature flat-global-insts}}
}

//--- store.cl
typedef __attribute__((__vector_size__(4 * sizeof(unsigned int)))) unsigned int v4u32;
typedef v4u32 __global *global_ptr_to_v4u32;

void test_amdgcn_av_store_b128_target(global_ptr_to_v4u32 ptr, v4u32 data) {
  __builtin_amdgcn_av_store_b128(ptr, data, 0); // expected-error{{'__builtin_amdgcn_av_store_b128' needs target feature flat-global-insts}}
}
