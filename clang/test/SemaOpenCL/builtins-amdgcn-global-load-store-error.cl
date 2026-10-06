// RUN: %clang_cc1 -cl-std=CL2.0 -triple amdgpu9.50-unknown-unknown         -S -verify -o - %s
// REQUIRES: amdgpu-registered-target

typedef unsigned short u16;
typedef unsigned int u32;
typedef unsigned long u64;

typedef __attribute__((__vector_size__(4 * sizeof(unsigned int)))) unsigned int v4u32;
typedef v4u32 __global *global_ptr_to_v4u32;
typedef v4u32 __private *private_ptr_to_v4u32;
typedef u16 __global *global_ptr_to_u16;
typedef u16 __private *private_ptr_to_u16;
typedef u32 __global *global_ptr_to_u32;
typedef u32 __private *private_ptr_to_u32;
typedef u64 __global *global_ptr_to_u64;
typedef u64 __private *private_ptr_to_u64;

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

void test_amdgcn_av_store_b16_bad_ptr(private_ptr_to_u16 ptr, u16 data) {
  __builtin_amdgcn_av_store_b16(ptr, data, __MEMORY_SCOPE_SYSTEM);  //expected-error{{builtin requires a global or generic pointer}}
}

void test_amdgcn_av_store_b16_bad_scope(global_ptr_to_u16 ptr, u16 data) {
  __builtin_amdgcn_av_store_b16(ptr, data, 42);  //expected-error{{synchronization scope argument to atomic operation is invalid}}
}

u16 test_amdgcn_av_load_b16_bad_ptr(private_ptr_to_u16 ptr) {
  return __builtin_amdgcn_av_load_b16(ptr, __MEMORY_SCOPE_SYSTEM);  //expected-error{{builtin requires a global or generic pointer}}
}

u16 test_amdgcn_av_load_b16_bad_scope(global_ptr_to_u16 ptr) {
  return __builtin_amdgcn_av_load_b16(ptr, 42);  //expected-error{{synchronization scope argument to atomic operation is invalid}}
}

void test_amdgcn_av_store_b32_bad_ptr(private_ptr_to_u32 ptr, u32 data) {
  __builtin_amdgcn_av_store_b32(ptr, data, __MEMORY_SCOPE_SYSTEM);  //expected-error{{builtin requires a global or generic pointer}}
}

void test_amdgcn_av_store_b32_bad_scope(global_ptr_to_u32 ptr, u32 data) {
  __builtin_amdgcn_av_store_b32(ptr, data, 42);  //expected-error{{synchronization scope argument to atomic operation is invalid}}
}

u32 test_amdgcn_av_load_b32_bad_ptr(private_ptr_to_u32 ptr) {
  return __builtin_amdgcn_av_load_b32(ptr, __MEMORY_SCOPE_SYSTEM);  //expected-error{{builtin requires a global or generic pointer}}
}

u32 test_amdgcn_av_load_b32_bad_scope(global_ptr_to_u32 ptr) {
  return __builtin_amdgcn_av_load_b32(ptr, 42);  //expected-error{{synchronization scope argument to atomic operation is invalid}}
}

void test_amdgcn_av_store_b64_bad_ptr(private_ptr_to_u64 ptr, u64 data) {
  __builtin_amdgcn_av_store_b64(ptr, data, __MEMORY_SCOPE_SYSTEM);  //expected-error{{builtin requires a global or generic pointer}}
}

void test_amdgcn_av_store_b64_bad_scope(global_ptr_to_u64 ptr, u64 data) {
  __builtin_amdgcn_av_store_b64(ptr, data, 42);  //expected-error{{synchronization scope argument to atomic operation is invalid}}
}

u64 test_amdgcn_av_load_b64_bad_ptr(private_ptr_to_u64 ptr) {
  return __builtin_amdgcn_av_load_b64(ptr, __MEMORY_SCOPE_SYSTEM);  //expected-error{{builtin requires a global or generic pointer}}
}

u64 test_amdgcn_av_load_b64_bad_scope(global_ptr_to_u64 ptr) {
  return __builtin_amdgcn_av_load_b64(ptr, 42);  //expected-error{{synchronization scope argument to atomic operation is invalid}}
}
