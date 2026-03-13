// REQUIRES: aarch64-registered-target

// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +sme -target-feature +sme2 -target-feature +bf16 -verify -emit-llvm -o - %s

#include <arm_sme.h>

svint8_t missing_sme2p3_zt(svuint8_t indices) __arm_streaming __arm_in("zt0") {
  return svluti6_zt_s8(0, indices); // expected-error {{'svluti6_zt_s8' needs target feature sme,sme2p3}}
}

__attribute__((target("sme2p3")))
svint8_t has_sme2p3_zt(svuint8_t indices) __arm_streaming __arm_in("zt0") {
  return svluti6_zt_s8(0, indices);
}

__attribute__((target("sme2p3")))
svfloat16_t has_sme2p3_implied_sme2p2(svbool_t pg, svfloat16_t op)
    __arm_streaming {
  return svcompact_f16(pg, op);
}
