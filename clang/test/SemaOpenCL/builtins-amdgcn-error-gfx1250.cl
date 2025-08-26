// REQUIRES: amdgpu-registered-target
// RUN: %clang_cc1 -triple amdgcn-- -target-cpu gfx1200 -verify -S -o - %s

void test() {
  __builtin_amdgcn_s_setprio_inc_wg(1); // expected-error {{'__builtin_amdgcn_s_setprio_inc_wg' needs target feature setprio-inc-wg-inst}}
}
