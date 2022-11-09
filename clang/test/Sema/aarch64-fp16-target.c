// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +neon -fsyntax-only -verify -emit-llvm -o - %s
// REQUIRES: aarch64-registered-target

// Test that functions with the correct target attributes can use the correct FP16 intrinsics.

#include <arm_fp16.h>

__attribute__((target("fullfp16")))
void test_fullfp16(float16_t f16) {
  vabdh_f16(f16, f16);
}

__attribute__((target("fp16")))
void fp16(float16_t f16) {
  vabdh_f16(f16, f16);
}

__attribute__((target("arch=armv8-a+fp16")))
void test_fp16_arch(float16_t f16) {
    vabdh_f16(f16, f16);
}

__attribute__((target("+fp16")))
void test_plusfp16(float16_t f16) {
    vabdh_f16(f16, f16);
}

void undefined(float16_t f16) {
  vabdh_f16(f16, f16); // expected-error {{'__builtin_neon_vabdh_f16' needs target feature fullfp16}}
}
