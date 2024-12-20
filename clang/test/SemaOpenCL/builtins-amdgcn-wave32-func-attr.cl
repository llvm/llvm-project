// RUN: %clang_cc1 -triple amdgcn-- -verify=default -S -o - %s
// RUN: %clang_cc1 -triple amdgcn-- -target-cpu gfx900 -verify=gfx9 -S -o - %s
// RUN: %clang_cc1 -triple amdgcn-- -target-cpu gfx1010 -verify=gfx10 -S -o - %s
// RUN: not %clang_cc1 -triple amdgcn-- -target-cpu gfx900 -target-feature -wavefrontsize32 -S -o - %s 2>&1 | FileCheck --check-prefix=GFX9 %s
// RUN: %clang_cc1 -triple amdgcn-- -target-cpu gfx1010 -target-feature -wavefrontsize32 -verify=gfx10 -S -o - %s

// REQUIRES: amdgpu-registered-target

// default-no-diagnostics
// gfx10-no-diagnostics

typedef unsigned int uint;

// GFX9: error: option 'wavefrontsize32' cannot be specified on this target
__attribute__((target("wavefrontsize32"))) // gfx9-error@*:* {{option 'wavefrontsize32' cannot be specified on this target}}
void test_ballot_wave32_target_attr(global uint* out, int a, int b) {
  *out = __builtin_amdgcn_ballot_w32(a == b);
}
