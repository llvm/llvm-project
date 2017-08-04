// REQUIRES: amdgpu-registered-target
// RUN: %clang_cc1 -triple amdgcn-- -target-cpu fiji -verify -S -o - %s

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

void test_gfx9_fmed3h(global half *out, half a, half b, half c)
{
  *out = __builtin_amdgcn_fmed3h(a, b, c); // expected-error {{'__builtin_amdgcn_fmed3h' needs target feature gfx9-insts}}
}
