// RUN: %clang_cc1 -cl-std=CL2.0 -triple amdgcn-unknown-unknown -target-cpu gfx950         -S -verify -o - %s
// REQUIRES: amdgpu-registered-target

typedef __attribute__((__vector_size__(4 * sizeof(unsigned int)))) unsigned int v4u32;
typedef v4u32 __global *global_ptr_to_v4u32;
typedef v4u32 __private *private_ptr_to_v4u32;

void test_amdgcn_av_store_b128_bad_ptr(private_ptr_to_v4u32 ptr, v4u32 data) {
  __builtin_amdgcn_av_store_b128(ptr, data, __MEMORY_SCOPE_SYSTEM);  //expected-error{{builtin requires a global or generic pointer}}
}

void test_amdgcn_av_store_b128_bad_scope(global_ptr_to_v4u32 ptr, v4u32 data) {
  __builtin_amdgcn_av_store_b128(ptr, data, 42);  //expected-error{{synchronization scope argument to atomic operation is invalid}}
}

v4u32 test_amdgcn_av_load_b128_bad_ptr(private_ptr_to_v4u32 ptr) {
  return __builtin_amdgcn_av_load_b128(ptr, __MEMORY_SCOPE_SYSTEM);  //expected-error{{builtin requires a global or generic pointer}}
}

v4u32 test_amdgcn_av_load_b128_bad_scope(global_ptr_to_v4u32 ptr) {
  return __builtin_amdgcn_av_load_b128(ptr, 42);  //expected-error{{synchronization scope argument to atomic operation is invalid}}
}
