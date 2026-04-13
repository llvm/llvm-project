// REQUIRES: aarch64-registered-target

// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +sve -verify -emit-llvm -o - %s

#include <arm_sve.h>

svfloat16_t missing_sve2p3_luti6_lane(svfloat16x2_t table, svuint8_t indices) {
  return svluti6_lane_f16_x2(table, indices, 1); // expected-error {{'svluti6_lane_f16_x2' needs target feature (sve,sve2p3)|(sme,(sve2p3|sme2p3))}}
}

__attribute__((target("sve2p3")))
svfloat16_t has_sve2p3_luti6_lane(svfloat16x2_t table, svuint8_t indices) {
  return svluti6_lane_f16_x2(table, indices, 0);
}

__attribute__((target("sve2p3,bf16")))
svbfloat16_t has_sve2p3_luti6_lane_bf16(svbfloat16x2_t table,
                                        svuint8_t indices) {
  return svluti6_lane_bf16_x2(table, indices, 1);
}

__attribute__((target("sve2p3,sme")))
svfloat16_t has_streaming_sve2p3_luti6_lane(svfloat16x2_t table,
                                            svuint8_t indices)
    __arm_streaming {
  return svluti6_lane_f16_x2(table, indices, 1);
}

__attribute__((target("sve2p3,sme,bf16")))
svbfloat16_t has_streaming_sve2p3_luti6_lane_bf16(svbfloat16x2_t table,
                                                  svuint8_t indices)
    __arm_streaming {
  return svluti6_lane_bf16_x2(table, indices, 0);
}

__attribute__((target("sme2p3,sme")))
svfloat16_t has_streaming_sme2p3_luti6_lane(svfloat16x2_t table,
                                            svuint8_t indices)
    __arm_streaming {
  return svluti6_lane_f16_x2(table, indices, 0);
}

__attribute__((target("sme2p3,sme,bf16")))
svbfloat16_t has_streaming_sme2p3_luti6_lane_bf16(svbfloat16x2_t table,
                                                  svuint8_t indices)
    __arm_streaming {
  return svluti6_lane_bf16_x2(table, indices, 1);
}
