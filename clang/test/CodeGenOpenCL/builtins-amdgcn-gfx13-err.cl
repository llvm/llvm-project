// REQUIRES: amdgpu-registered-target

// RUN: %clang_cc1 -cl-std=CL2.0 -triple amdgcn-unknown-unknown -target-cpu gfx1300 -verify -emit-llvm -o - %s

void test_setprio_inc_wg() {
  __builtin_amdgcn_s_setprio_inc_wg(10); // expected-error {{'__builtin_amdgcn_s_setprio_inc_wg' needs target feature setprio-inc-wg-inst}}
}

void test_s_inst_auto_prefetch_mode(short a) {
  __builtin_amdgcn_s_inst_auto_prefetch_mode(a); // expected-error {{'__builtin_amdgcn_s_inst_auto_prefetch_mode' must be a constant integer}}
}
