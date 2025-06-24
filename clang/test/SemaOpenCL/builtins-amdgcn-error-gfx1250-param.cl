// REQUIRES: amdgpu-registered-target
// RUN: %clang_cc1 -triple amdgcn-- -target-cpu gfx1250 -verify -S -o - %s

void test_setprio_inc_wg(short a) {
  __builtin_amdgcn_s_setprio_inc_wg(a); // expected-error {{'__builtin_amdgcn_s_setprio_inc_wg' must be a constant integer}}
}
