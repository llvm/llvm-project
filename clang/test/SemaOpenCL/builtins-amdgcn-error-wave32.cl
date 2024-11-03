// RUN: %clang_cc1 -triple amdgcn-- -verify -S -o - %s
// RUN: %clang_cc1 -triple amdgcn-- -target-cpu gfx900 -verify -S -o - %s
// RUN: %clang_cc1 -triple amdgcn-- -target-cpu gfx900 -target-feature +wavefrontsize64 -verify -S -o - %s
// RUN: %clang_cc1 -triple amdgcn-- -target-cpu gfx1010 -target-feature +wavefrontsize64 -verify -S -o - %s
// RUN: %clang_cc1 -triple amdgcn-- -target-cpu gfx1010 -target-feature -wavefrontsize32 -verify -S -o - %s

typedef unsigned int uint;

void test_ballot_wave32(global uint* out, int a, int b) {
  *out = __builtin_amdgcn_ballot_w32(a == b);  // expected-error {{'__builtin_amdgcn_ballot_w32' needs target feature wavefrontsize32}}
}

// FIXME: Should error for subtargets that don't support wave32
__attribute__((target("wavefrontsize32")))
void test_ballot_wave32_target_attr(global uint* out, int a, int b) {
  *out = __builtin_amdgcn_ballot_w32(a == b);
}
