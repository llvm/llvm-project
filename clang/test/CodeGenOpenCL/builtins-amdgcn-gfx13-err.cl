// REQUIRES: amdgpu-registered-target

// RUN: %clang_cc1 -cl-std=CL2.0 -triple amdgcn-unknown-unknown -target-cpu gfx1300 -verify -emit-llvm -o - %s

void test_s_inst_auto_prefetch_mode(short a) {
  __builtin_amdgcn_s_inst_auto_prefetch_mode(a); // expected-error {{'__builtin_amdgcn_s_inst_auto_prefetch_mode' must be a constant integer}}
}
