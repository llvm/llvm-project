// REQUIRES: amdgpu-registered-target

// RUN: %clang_cc1 -triple amdgcn-unknown-unknown -target-cpu gfx1200 -verify -S -emit-llvm -o - %s

typedef unsigned int uint;

kernel void test_builtins_amdgcn_gws_insts(uint a, uint b) {
  __builtin_amdgcn_ds_gws_init(a, b); // expected-error {{'__builtin_amdgcn_ds_gws_init' needs target feature gws}}
  __builtin_amdgcn_ds_gws_barrier(a, b); // expected-error {{'__builtin_amdgcn_ds_gws_barrier' needs target feature gws}}
  __builtin_amdgcn_ds_gws_sema_v(a); // expected-error {{'__builtin_amdgcn_ds_gws_sema_v' needs target feature gws}}
  __builtin_amdgcn_ds_gws_sema_br(a, b); // expected-error {{'__builtin_amdgcn_ds_gws_sema_br' needs target feature gws}}
  __builtin_amdgcn_ds_gws_sema_p(a); // expected-error {{'__builtin_amdgcn_ds_gws_sema_p' needs target feature gws}}
}
