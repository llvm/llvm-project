// REQUIRES: amdgpu-registered-target
// RUN: %clang_cc1 -triple amdgcn-- -target-cpu tahiti -verify -S -o - %s
// RUN: %clang_cc1 -triple amdgcn-- -target-cpu hawaii -verify -S -o - %s
// RUN: %clang_cc1 -triple amdgcn-- -target-cpu fiji -verify -S -o - %s
// RUN: %clang_cc1 -triple amdgcn-- -target-cpu gfx900 -verify -S -o - %s
// RUN: %clang_cc1 -triple amdgcn-- -target-cpu gfx908 -verify -S -o - %s

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

typedef unsigned int uint;
typedef int int2 __attribute__((ext_vector_type(2)));

struct S {
  int x;
};

void test(global uint* out, uint a, uint b, uint c, uint d) {
  *out = __builtin_amdgcn_permlane16(a, b, c, d, 1, 1); // expected-error {{'__builtin_amdgcn_permlane16' needs target feature gfx10-insts}}
  *out = __builtin_amdgcn_permlanex16(a, b, c, d, 1, 1);  // expected-error {{'__builtin_amdgcn_permlanex16' needs target feature gfx10-insts}}
  *out = __builtin_amdgcn_mov_dpp8(a, 1);  // expected-error {{'__builtin_amdgcn_mov_dpp8' needs target feature gfx10-insts}}
}

void test_mov_dpp8(global int* out, int src, int i, int2 i2, struct S s, float _Complex fc)
{
  *out = __builtin_amdgcn_mov_dpp8(src, i); // expected-error{{argument to '__builtin_amdgcn_mov_dpp8' must be a constant integer}}
  *out = __builtin_amdgcn_mov_dpp8(src, 0.1); // expected-error{{argument to '__builtin_amdgcn_mov_dpp8' must be a constant integer}}
  *out = __builtin_amdgcn_mov_dpp8(src); // expected-error{{too few arguments to function call, expected 2, have 1}}
  *out = __builtin_amdgcn_mov_dpp8(src, 0, 0); // expected-error{{too many arguments to function call, expected at most 2, have 3}}
  *out = __builtin_amdgcn_mov_dpp8(out, 0); // expected-error{{used type '__global int *__private' where integer or floating point type is required}}
  *out = __builtin_amdgcn_mov_dpp8("aa", 0); // expected-error{{used type '__constant char[3]' where integer or floating point type is required}}
  *out = __builtin_amdgcn_mov_dpp8(i2, 0); // expected-error{{used type '__private int2' (vector of 2 'int' values) where integer or floating point type is required}}
  *out = __builtin_amdgcn_mov_dpp8(s, 0); // expected-error{{used type '__private struct S' where integer or floating point type is required}}
  *out = __builtin_amdgcn_mov_dpp8(fc, 0); // expected-error{{used type '__private _Complex float' where integer or floating point type is required}}
}
