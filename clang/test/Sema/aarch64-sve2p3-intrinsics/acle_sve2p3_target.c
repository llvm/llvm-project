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

__attribute__((target("sve2p3")))
svfloat32_t has_sve2p3_implied_sve2p2(svbool_t pg, svfloat16_t op) {
  return svcvtlt_f32_f16_z(pg, op);
}
