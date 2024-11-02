// REQUIRES: amdgpu-registered-target

// RUN: %clang_cc1 -triple amdgcn-unknown-unknown -target-cpu gfx1100 -verify -S -emit-llvm -o - %s

void test_s_sleep_var(int d)
{
  __builtin_amdgcn_s_sleep_var(d); // expected-error {{'__builtin_amdgcn_s_sleep_var' needs target feature gfx12-insts}}
}
