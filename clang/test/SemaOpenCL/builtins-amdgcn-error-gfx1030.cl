// REQUIRES: amdgpu-registered-target
// RUN: %clang_cc1 -triple amdgcn-- -target-cpu gfx1030 -verify -S -o - %s

void test_gfx1030_s_memtime()
{
  __builtin_amdgcn_s_memtime(); // expected-error {{'__builtin_amdgcn_s_memtime' needs target feature s-memtime-inst}}
}
