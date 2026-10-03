// REQUIRES: amdgpu-registered-target
// RUN: %clang_cc1 -triple amdgpu6.00-- -verify -S -o - %s
// RUN: %clang_cc1 -triple amdgpu7.01-- -verify -S -o - %s

void test_vi_builtins()
{
  __builtin_amdgcn_s_dcache_wb(); // expected-error {{'__builtin_amdgcn_s_dcache_wb' needs target feature gfx8-insts}}
  (void)__builtin_amdgcn_perm(1, 2, 3); // expected-error {{'__builtin_amdgcn_perm' needs target feature gfx8-insts}}
}
