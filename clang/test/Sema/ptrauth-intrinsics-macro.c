// RUN: %clang_cc1 -triple arm64-apple-ios -fsyntax-only -verify -fptrauth-intrinsics %s
// RUN: %clang_cc1 -triple arm64-apple-ios -fsyntax-only -verify %s

// expected-no-diagnostics

#include <ptrauth.h>

#define VALID_CODE_KEY 0
#define VALID_DATA_KEY 2

extern int dv;

void test(int *dp, int (*fp)(int), int value) {
  dp = ptrauth_strip(dp, VALID_DATA_KEY);
  uintptr_t t0 = ptrauth_blend_discriminator(dp, value);
  t0 = ptrauth_type_discriminator(int (*)(int));
  dp = ptrauth_sign_constant(&dv, VALID_DATA_KEY, 0);
  dp = ptrauth_sign_unauthenticated(dp, VALID_DATA_KEY, 0);
  dp = ptrauth_auth_and_resign(dp, VALID_DATA_KEY, dp, VALID_DATA_KEY, dp);
  fp = ptrauth_auth_function(fp, VALID_CODE_KEY, 0);
  dp = ptrauth_auth_data(dp, VALID_DATA_KEY, 0);
  int t1 = ptrauth_string_discriminator("string");
  int t2 = ptrauth_sign_generic_data(dp, 0);

  void * __ptrauth_function_pointer p0;
  void * __ptrauth_return_address p1;
  void * __ptrauth_block_invocation_pointer p2;
  void * __ptrauth_block_copy_helper p3;
  void * __ptrauth_block_destroy_helper p4;
  void * __ptrauth_block_byref_copy_helper p5;
  void * __ptrauth_block_byref_destroy_helper p6;
  void * __ptrauth_objc_method_list_imp p7;
  void * __ptrauth_cxx_vtable_pointer p8;
  void * __ptrauth_cxx_vtt_vtable_pointer p9;
  void * __ptrauth_swift_heap_object_destructor p10;
  void * __ptrauth_swift_function_pointer(VALID_CODE_KEY) p11;
  void * __ptrauth_swift_class_method_pointer(VALID_CODE_KEY) p12;
  void * __ptrauth_swift_protocol_witness_function_pointer(VALID_CODE_KEY) p13;
  void * __ptrauth_swift_value_witness_function_pointer(VALID_CODE_KEY) p14;
}
