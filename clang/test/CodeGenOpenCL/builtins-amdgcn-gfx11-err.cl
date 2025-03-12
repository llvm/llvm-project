// REQUIRES: amdgpu-registered-target

// RUN: %clang_cc1 -triple amdgcn-unknown-unknown -target-cpu gfx1100 -verify -emit-llvm -o - %s

void builtin_test_unsupported(int a, int b) {
  __builtin_amdgcn_s_sleep_var(a); // expected-error {{'__builtin_amdgcn_s_sleep_var' needs target feature gfx12-insts}}
  b = __builtin_amdgcn_ds_bpermute_fi_b32(a, b); // expected-error {{'__builtin_amdgcn_ds_bpermute_fi_b32' needs target feature gfx12-insts}}
}
