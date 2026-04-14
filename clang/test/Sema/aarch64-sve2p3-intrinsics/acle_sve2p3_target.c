// REQUIRES: aarch64-registered-target

// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +sve -verify -emit-llvm -o - %s

#include <arm_sve.h>

void missing_sve2p3_luti6(svint8x2_t table, svuint8_t indices) {
  svluti6_s8(table, indices); // expected-error {{'svluti6_s8' needs target feature sve,sve2p3}}
}

__attribute__((target("sve2p3")))
svint8_t has_sve2p3_luti6(svint8x2_t table, svuint8_t indices) {
  return svluti6_s8(table, indices);
}

__attribute__((target("sve2p3,bf16")))
void has_sve2p3_luti6_lane_bf16(svfloat16x2_t f16_table,
                                svbfloat16x2_t bf16_table,
                                svuint8_t indices) {
  (void)svluti6_lane_f16_x2(f16_table, indices, 0);
  (void)svluti6_lane_bf16_x2(bf16_table, indices, 1);
}

__attribute__((target("sve2p3,sme")))
svfloat16_t has_streaming_sve2p3_luti6_lane(svfloat16x2_t table,
                                            svuint8_t indices)
    __arm_streaming {
  return svluti6_lane_f16_x2(table, indices, 1);
}

__attribute__((target("sve2p3,sme,bf16")))
void has_streaming_sve2p3_luti6_lane_bf16(svfloat16x2_t f16_table,
                                          svbfloat16x2_t bf16_table,
                                          svuint8_t indices)
    __arm_streaming {
  (void)svluti6_lane_f16_x2(f16_table, indices, 1);
  (void)svluti6_lane_bf16_x2(bf16_table, indices, 0);
}

__attribute__((target("sme2p3,sme")))
svfloat16_t has_streaming_sme2p3_luti6_lane(svfloat16x2_t table,
                                            svuint8_t indices)
    __arm_streaming {
  return svluti6_lane_f16_x2(table, indices, 0);
}

__attribute__((target("sme2p3,sme,bf16")))
void has_streaming_sme2p3_luti6_lane_bf16(svfloat16x2_t f16_table,
                                          svbfloat16x2_t bf16_table,
                                          svuint8_t indices)
    __arm_streaming {
  (void)svluti6_lane_f16_x2(f16_table, indices, 0);
  (void)svluti6_lane_bf16_x2(bf16_table, indices, 1);
}
