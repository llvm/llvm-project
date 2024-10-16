// REQUIRES: amdgpu-registered-target
// RUN: %clang_cc1 -triple amdgcn-- -target-cpu fiji -verify -S -o - %s

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

void test_gfx9_fmed3h(global half *out, half a, half b, half c)
{
  *out = __builtin_amdgcn_fmed3h(a, b, c); // expected-error {{'__builtin_amdgcn_fmed3h' needs target feature gfx9-insts}}
}

void test_mov_dpp(global int* out, int src, int i)
{
  *out = __builtin_amdgcn_mov_dpp(src, i, 0, 0, false); // expected-error{{argument to '__builtin_amdgcn_mov_dpp' must be a constant integer}}
  *out = __builtin_amdgcn_mov_dpp(src, 0, i, 0, false); // expected-error{{argument to '__builtin_amdgcn_mov_dpp' must be a constant integer}}
  *out = __builtin_amdgcn_mov_dpp(src, 0, 0, i, false); // expected-error{{argument to '__builtin_amdgcn_mov_dpp' must be a constant integer}}
  *out = __builtin_amdgcn_mov_dpp(src, 0, 0, 0, i); // expected-error{{argument to '__builtin_amdgcn_mov_dpp' must be a constant integer}}
  *out = __builtin_amdgcn_mov_dpp(src, 0.1, 0, 0, false); // expected-error{{argument to '__builtin_amdgcn_mov_dpp' must be a constant integer}}
  *out = __builtin_amdgcn_mov_dpp(src, 0, 0.1, 0, false); // expected-error{{argument to '__builtin_amdgcn_mov_dpp' must be a constant integer}}
  *out = __builtin_amdgcn_mov_dpp(src, 0, 0, 0.1, false); // expected-error{{argument to '__builtin_amdgcn_mov_dpp' must be a constant integer}}
  *out = __builtin_amdgcn_mov_dpp(src, 0, 0, 0, 0.1); // expected-error{{argument to '__builtin_amdgcn_mov_dpp' must be a constant integer}}
  *out = __builtin_amdgcn_mov_dpp(src, 0, 0, 0); // expected-error{{too few arguments to function call, expected 5, have 4}}
  *out = __builtin_amdgcn_mov_dpp(src, 0, 0, 0, false, 1); // expected-error{{too many arguments to function call, expected at most 5, have 6}}
  *out = __builtin_amdgcn_mov_dpp(out, 0, 0, 0, false); // expected-error{{used type '__global int *__private' where integer or floating point type is required}}
  *out = __builtin_amdgcn_mov_dpp("aa", 0, 0, 0, false); // expected-error{{used type '__constant char[3]' where integer or floating point type is required}}
}

void test_update_dpp(global int* out, int arg1, int arg2, int i)
{
  *out = __builtin_amdgcn_update_dpp(arg1, arg2, i, 0, 0, false); // expected-error{{argument to '__builtin_amdgcn_update_dpp' must be a constant integer}}
  *out = __builtin_amdgcn_update_dpp(arg1, arg2, 0, i, 0, false); // expected-error{{argument to '__builtin_amdgcn_update_dpp' must be a constant integer}}
  *out = __builtin_amdgcn_update_dpp(arg1, arg2, 0, 0, i, false); // expected-error{{argument to '__builtin_amdgcn_update_dpp' must be a constant integer}}
  *out = __builtin_amdgcn_update_dpp(arg1, arg2, 0, 0, 0, i); // expected-error{{argument to '__builtin_amdgcn_update_dpp' must be a constant integer}}
  *out = __builtin_amdgcn_update_dpp(arg1, arg2, 0.1, 0, 0, false); // expected-error{{argument to '__builtin_amdgcn_update_dpp' must be a constant integer}}
  *out = __builtin_amdgcn_update_dpp(arg1, arg2, 0, 0.1, 0, false); // expected-error{{argument to '__builtin_amdgcn_update_dpp' must be a constant integer}}
  *out = __builtin_amdgcn_update_dpp(arg1, arg2, 0, 0, 0.1, false); // expected-error{{argument to '__builtin_amdgcn_update_dpp' must be a constant integer}}
  *out = __builtin_amdgcn_update_dpp(arg1, arg2, 0, 0, 0, 0.1); // expected-error{{argument to '__builtin_amdgcn_update_dpp' must be a constant integer}}
  *out = __builtin_amdgcn_update_dpp(arg1, arg2, 0, 0, 0); // expected-error{{too few arguments to function call, expected 6, have 5}}
  *out = __builtin_amdgcn_update_dpp(arg1, arg2, 0, 0, 0, false, 1); // expected-error{{too many arguments to function call, expected at most 6, have 7}}
  *out = __builtin_amdgcn_update_dpp(out, arg2, 0, 0, 0, false); // expected-error{{used type '__global int *__private' where integer or floating point type is required}}
  *out = __builtin_amdgcn_update_dpp(arg1, out, 0, 0, 0, false); // expected-error{{used type '__global int *__private' where integer or floating point type is required}}
  *out = __builtin_amdgcn_update_dpp("aa", arg2, 0, 0, 0, false); // expected-error{{used type '__constant char[3]' where integer or floating point type is required}}
  *out = __builtin_amdgcn_update_dpp(arg1, "aa", 0, 0, 0, false); // expected-error{{used type '__constant char[3]' where integer or floating point type is required}}
}
