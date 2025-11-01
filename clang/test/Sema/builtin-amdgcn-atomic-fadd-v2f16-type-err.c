// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -verify %s

// Test semantic analysis for AMDGCN atomic fadd v2f16 builtins
// These tests ensure proper type checking for the builtin arguments

typedef _Float16 v2f16 __attribute__((ext_vector_type(2)));
typedef float v2f32 __attribute__((ext_vector_type(2)));
typedef _Float16 v4f16 __attribute__((ext_vector_type(4)));

void test_global_atomic_fadd_v2f16_negative() {
  v2f16 val;
  v2f32 val_f32;
  v4f16 val_v4f16;
  _Float16 __attribute__((ext_vector_type(2))) __attribute__((address_space(1))) *ptr_v2f16;

  __builtin_amdgcn_global_atomic_fadd_v2f16(ptr_v2f16, val_f32); // expected-error{{passing 'v2f32'}}
  __builtin_amdgcn_global_atomic_fadd_v2f16(ptr_v2f16, val_v4f16); // expected-error{{passing 'v4f16'}}
  __builtin_amdgcn_global_atomic_fadd_v2f16(ptr_v2f16); // expected-error{{too few arguments to function call}}
  __builtin_amdgcn_global_atomic_fadd_v2f16(ptr_v2f16, val, val); // expected-error{{too many arguments to function call}}
}

void test_flat_atomic_fadd_v2f16_negative() {
  v2f16 val;
  v2f32 val_f32;
  v4f16 val_v4f16;
  _Float16 __attribute__((ext_vector_type(2)))  __attribute__((address_space(0))) *ptr_v2f16;

  // Same error patterns for flat atomic
  __builtin_amdgcn_flat_atomic_fadd_v2f16(ptr_v2f16, val_f32); // expected-error{{passing 'v2f32'}}
  __builtin_amdgcn_flat_atomic_fadd_v2f16(ptr_v2f16, val_v4f16); // expected-error{{passing 'v4f16'}}
  __builtin_amdgcn_flat_atomic_fadd_v2f16(ptr_v2f16); // expected-error{{too few arguments to function call}}
  __builtin_amdgcn_flat_atomic_fadd_v2f16(ptr_v2f16, val, val); // expected-error{{too many arguments to function call}}
}

void test_ds_atomic_fadd_v2f16_negative() {
  v2f16 val;
  v2f32 val_f32;
  v4f16 val_v4f16;
  _Float16 __attribute__((ext_vector_type(2)))  __attribute__((address_space(3))) *ptr_v2f16;

  // Same error patterns for ds atomic
  __builtin_amdgcn_ds_atomic_fadd_v2f16(ptr_v2f16, val_f32); // expected-error{{passing 'v2f32'}}
  __builtin_amdgcn_ds_atomic_fadd_v2f16(ptr_v2f16, val_v4f16); // expected-error{{passing 'v4f16'}}
  __builtin_amdgcn_ds_atomic_fadd_v2f16(ptr_v2f16); // expected-error{{too few arguments to function call}}
  __builtin_amdgcn_ds_atomic_fadd_v2f16(ptr_v2f16, val, val); // expected-error{{too many arguments to function call}}
}
