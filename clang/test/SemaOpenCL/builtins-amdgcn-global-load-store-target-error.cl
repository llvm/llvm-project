// We test loads and stores separately because clang only seems to exit after
// the first 'target feature' error.

// RUN: %clang_cc1 -cl-std=CL2.0 -triple amdgcn-unknown-unknown -target-cpu gfx602 -DTEST_LOAD  -S -verify -o - %s
// RUN: %clang_cc1 -cl-std=CL2.0 -triple amdgcn-unknown-unknown -target-cpu gfx705 -DTEST_LOAD  -S -verify -o - %s
// RUN: %clang_cc1 -cl-std=CL2.0 -triple amdgcn-unknown-unknown -target-cpu gfx810 -DTEST_LOAD  -S -verify -o - %s

// RUN: %clang_cc1 -cl-std=CL2.0 -triple amdgcn-unknown-unknown -target-cpu gfx602 -DTEST_STORE -S -verify -o - %s
// RUN: %clang_cc1 -cl-std=CL2.0 -triple amdgcn-unknown-unknown -target-cpu gfx705 -DTEST_STORE -S -verify -o - %s
// RUN: %clang_cc1 -cl-std=CL2.0 -triple amdgcn-unknown-unknown -target-cpu gfx810 -DTEST_STORE -S -verify -o - %s
// REQUIRES: amdgpu-registered-target

typedef __attribute__((__vector_size__(4 * sizeof(unsigned int)))) unsigned int v4u32;
typedef v4u32 __global *global_ptr_to_v4u32;

#ifdef TEST_LOAD
v4u32 test_amdgcn_av_load_b128_target(global_ptr_to_v4u32 ptr) {
  return __builtin_amdgcn_av_load_b128(ptr, 0); // expected-error{{'__builtin_amdgcn_av_load_b128' needs target feature gfx9-insts}}
}
#endif

#ifdef TEST_STORE
void test_amdgcn_av_store_b128_target(global_ptr_to_v4u32 ptr, v4u32 data) {
  __builtin_amdgcn_av_store_b128(ptr, data, 0); // expected-error{{'__builtin_amdgcn_av_store_b128' needs target feature gfx9-insts}}
}
#endif
