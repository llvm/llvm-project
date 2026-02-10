// RUN: %clang_cc1 -triple amdgcn-unknown-unknown -target-cpu gfx950         -S -verify -o - %s
// RUN: %clang_cc1 -triple amdgcn-unknown-unknown -target-cpu gfx9-4-generic -S -verify -o - %s
// REQUIRES: amdgpu-registered-target

typedef __attribute__((__vector_size__(4 * sizeof(unsigned int)))) unsigned int v4u32;
typedef v4u32 __global *global_ptr_to_v4u32;

void test_amdgcn_global_store_b128_00(v4u32 *ptr, v4u32 data, const char* scope) {
  __builtin_amdgcn_global_store_b128(ptr, data, "");  //expected-error{{passing '__private v4u32 *__private' to parameter of type '__attribute__((__vector_size__(4 * sizeof(unsigned int)))) unsigned int __global *' changes address space of pointer}}
}

void test_amdgcn_global_store_b128_01(global_ptr_to_v4u32 ptr, v4u32 data, const char* scope) {
  __builtin_amdgcn_global_store_b128(ptr, data, scope);  //expected-error{{expression is not a string literal}}
}

v4u32 test_amdgcn_global_load_b128_00(v4u32 *ptr, const char* scope) {
  return __builtin_amdgcn_global_load_b128(ptr, "");  //expected-error{{passing '__private v4u32 *__private' to parameter of type '__attribute__((__vector_size__(4 * sizeof(unsigned int)))) unsigned int __global *' changes address space of pointer}}
}

v4u32 test_amdgcn_global_load_b128_01(global_ptr_to_v4u32 ptr, const char* scope) {
  return __builtin_amdgcn_global_load_b128(ptr, scope);  //expected-error{{expression is not a string literal}}
}
