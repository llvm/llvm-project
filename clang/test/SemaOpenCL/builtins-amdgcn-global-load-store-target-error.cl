// We test loads and stores separately because clang only seems to exit after
// the first 'target feature' error.

// RUN: %clang_cc1 -triple amdgcn-unknown-unknown -target-cpu gfx9-generic    -DTEST_LOAD  -S -verify -o - %s
// RUN: %clang_cc1 -triple amdgcn-unknown-unknown -target-cpu gfx10-1-generic -DTEST_LOAD  -S -verify -o - %s
// RUN: %clang_cc1 -triple amdgcn-unknown-unknown -target-cpu gfx10-3-generic -DTEST_LOAD  -S -verify -o - %s
// RUN: %clang_cc1 -triple amdgcn-unknown-unknown -target-cpu gfx11-generic   -DTEST_LOAD  -S -verify -o - %s
// RUN: %clang_cc1 -triple amdgcn-unknown-unknown -target-cpu gfx12-generic   -DTEST_LOAD  -S -verify -o - %s

// RUN: %clang_cc1 -triple amdgcn-unknown-unknown -target-cpu gfx9-generic    -DTEST_STORE -S -verify -o - %s
// RUN: %clang_cc1 -triple amdgcn-unknown-unknown -target-cpu gfx10-1-generic -DTEST_STORE -S -verify -o - %s
// RUN: %clang_cc1 -triple amdgcn-unknown-unknown -target-cpu gfx10-3-generic -DTEST_STORE -S -verify -o - %s
// RUN: %clang_cc1 -triple amdgcn-unknown-unknown -target-cpu gfx11-generic   -DTEST_STORE -S -verify -o - %s
// RUN: %clang_cc1 -triple amdgcn-unknown-unknown -target-cpu gfx12-generic   -DTEST_STORE -S -verify -o - %s
// REQUIRES: amdgpu-registered-target

typedef __attribute__((__vector_size__(4 * sizeof(unsigned int)))) unsigned int v4u32;
typedef v4u32 __global *global_ptr_to_v4u32;

#ifdef TEST_LOAD
v4u32 test_amdgcn_global_load_b128_01(global_ptr_to_v4u32 ptr, const char* scope) {
  return __builtin_amdgcn_global_load_b128(ptr, ""); // expected-error{{'__builtin_amdgcn_global_load_b128' needs target feature gfx940-insts}}
}
#endif

#ifdef TEST_STORE
void test_amdgcn_global_store_b128_01(global_ptr_to_v4u32 ptr, v4u32 data, const char* scope) {
  __builtin_amdgcn_global_store_b128(ptr, data, ""); // expected-error{{'__builtin_amdgcn_global_store_b128' needs target feature gfx940-insts}}
}
#endif
