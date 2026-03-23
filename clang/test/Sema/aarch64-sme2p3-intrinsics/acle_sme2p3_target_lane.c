// REQUIRES: aarch64-registered-target

// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +sme -target-feature +sme2 -target-feature +bf16 -verify -emit-llvm -o - %s

#include <arm_sme.h>

svbfloat16x4_t missing_sme2p3_lane(svbfloat16x2_t table, svuint8x2_t indices)
    __arm_streaming {
  return svluti6_lane_bf16_x4(table, indices, 1); // expected-error {{'svluti6_lane_bf16_x4' needs target feature sme,sme2p3}}
}

__attribute__((target("sme2p3,bf16")))
svbfloat16x4_t has_sme2p3_lane(svbfloat16x2_t table, svuint8x2_t indices)
    __arm_streaming {
  return svluti6_lane_bf16_x4(table, indices, 0);
}
