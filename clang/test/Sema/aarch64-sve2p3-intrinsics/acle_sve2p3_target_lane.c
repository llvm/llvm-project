// REQUIRES: aarch64-registered-target

// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +sve -verify -emit-llvm -o - %s

#include <arm_sve.h>

svfloat16_t missing_sve2p3_luti6_lane(svfloat16x2_t table, svuint8_t indices) {
  return svluti6_lane_f16_x2(table, indices, 1); // expected-error {{'svluti6_lane_f16_x2' needs target feature (sve,sve2p3)|(sme,sme2p3)}}
}

__attribute__((target("sve2p3")))
svfloat16_t has_sve2p3_luti6_lane(svfloat16x2_t table, svuint8_t indices) {
  return svluti6_lane_f16_x2(table, indices, 0);
}

svfloat16x4_t missing_sve2p3_luti6_lane_x4(svfloat16x2_t table, svuint8x2_t indices) {
  return svluti6_lane_f16_x4(table, indices, 1); // expected-error {{'svluti6_lane_f16_x4' needs target feature (sve,sve2p3)|(sme,sme2p3)}}
}

__attribute__((target("sve2p3")))
svfloat16x4_t has_sve2p3_luti6_lane_x4(svfloat16x2_t table, svuint8x2_t indices) {
  return svluti6_lane_f16_x4(table, indices, 0);
}
