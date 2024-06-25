// RUN: %clang_cc1 -triple aarch64 -target-feature +sme -verify -emit-llvm-only %s

// REQUIRES: aarch64-registered-target

#include <arm_sme.h>

void test_features(uint32_t slice, svfloat16x2_t zn2, svfloat16x4_t zn4,
                   svbfloat16x2_t bzn2, svbfloat16x4_t bzn4) __arm_streaming __arm_inout("za") {
  // expected-error@+1 {{'svadd_za16_f16_vg1x2' needs target feature sme-f16f16|sme-f8f16}}
  svadd_za16_f16_vg1x2(slice, zn2);
  // expected-error@+1 {{'svadd_za16_f16_vg1x4' needs target feature sme-f16f16|sme-f8f16}}
  svadd_za16_f16_vg1x4(slice, zn4);
  // expected-error@+1 {{'svsub_za16_f16_vg1x2' needs target feature sme-f16f16|sme-f8f16}}
  svsub_za16_f16_vg1x2(slice, zn2);
  // expected-error@+1 {{'svsub_za16_f16_vg1x4' needs target feature sme-f16f16|sme-f8f16}}
  svsub_za16_f16_vg1x4(slice, zn4);

  // expected-error@+1 {{'svadd_za16_bf16_vg1x2' needs target feature sme2,b16b16}}
  svadd_za16_bf16_vg1x2(slice, bzn2);
  // expected-error@+1 {{'svadd_za16_bf16_vg1x4' needs target feature sme2,b16b16}}
  svadd_za16_bf16_vg1x4(slice, bzn4);
  // expected-error@+1 {{'svsub_za16_bf16_vg1x2' needs target feature sme2,b16b16}}
  svsub_za16_bf16_vg1x2(slice, bzn2);
  // expected-error@+1 {{'svsub_za16_bf16_vg1x4' needs target feature sme2,b16b16}}
  svsub_za16_bf16_vg1x4(slice, bzn4);
}



