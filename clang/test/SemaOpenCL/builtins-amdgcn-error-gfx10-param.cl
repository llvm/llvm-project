// RUN: %clang_cc1 -triple amdgcn-- -target-cpu gfx1010 -verify -S -o - %s

typedef unsigned int uint;


void test_permlane16(global uint* out, uint a, uint b, uint c, uint d, uint e) {
  *out = __builtin_amdgcn_permlane16(a, b, c, d, e, 1); // expected-error{{argument to '__builtin_amdgcn_permlane16' must be a constant integer}}
  *out = __builtin_amdgcn_permlane16(a, b, c, d, 1, e); // expected-error{{argument to '__builtin_amdgcn_permlane16' must be a constant integer}}
}

void test_permlanex16(global uint* out, uint a, uint b, uint c, uint d, uint e) {
  *out = __builtin_amdgcn_permlanex16(a, b, c, d, e, 1); // expected-error{{argument to '__builtin_amdgcn_permlanex16' must be a constant integer}}
  *out = __builtin_amdgcn_permlanex16(a, b, c, d, 1, e); // expected-error{{argument to '__builtin_amdgcn_permlanex16' must be a constant integer}}
}

void test_mov_dpp8(global uint* out, uint a, uint b) {
  *out = __builtin_amdgcn_mov_dpp8(a, b); // expected-error{{argument to '__builtin_amdgcn_mov_dpp8' must be a constant integer}}
}
