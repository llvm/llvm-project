// RUN: %clang_cc1 -triple aarch64 -target-feature +sme -verify -emit-llvm-only %s

// REQUIRES: aarch64-registered-target

#include <arm_sme.h>

void test_svmla(uint32_t slice, svmfloat8_t zn, svmfloat8x2_t znx2, svmfloat8x4_t znx4,
                fpm_t fpmr) __arm_streaming __arm_inout("za") {
    // expected-error@+1 {{'svmla_lane_za16_mf8_vg2x1_fpm' needs target feature sme,sme-f8f16}}
    svmla_lane_za16_mf8_vg2x1_fpm(slice, zn, zn, 0, fpmr);

    // expected-error@+1 {{'svmla_lane_za16_mf8_vg2x2_fpm' needs target feature sme,sme-f8f16}}
    svmla_lane_za16_mf8_vg2x2_fpm(slice, znx2, zn, 0, fpmr);

    // expected-error@+1 {{'svmla_lane_za16_mf8_vg2x4_fpm' needs target feature sme,sme-f8f16}}
    svmla_lane_za16_mf8_vg2x4_fpm(slice, znx4, zn, 0, fpmr);

    // expected-error@+1 {{'svmla_lane_za32_mf8_vg4x1_fpm' needs target feature sme,sme-f8f32}}
    svmla_lane_za32_mf8_vg4x1_fpm(slice, zn, zn, 0, fpmr);

    // expected-error@+1 {{'svmla_lane_za32_mf8_vg4x2_fpm' needs target feature sme,sme-f8f32}}
    svmla_lane_za32_mf8_vg4x2_fpm(slice, znx2, zn, 0, fpmr);

    // expected-error@+1 {{'svmla_lane_za32_mf8_vg4x4_fpm' needs target feature sme,sme-f8f32}}
    svmla_lane_za32_mf8_vg4x4_fpm(slice, znx4, zn, 0, fpmr);

    // expected-error@+1 {{'svmla_single_za16_mf8_vg2x1_fpm' needs target feature sme,sme-f8f16}}
    svmla_single_za16_mf8_vg2x1_fpm(slice, zn, zn, fpmr);

    // expected-error@+1 {{'svmla_single_za16_mf8_vg2x2_fpm' needs target feature sme,sme-f8f16}}
    svmla_single_za16_mf8_vg2x2_fpm(slice, znx2, zn, fpmr);

    // expected-error@+1 {{'svmla_single_za16_mf8_vg2x4_fpm' needs target feature sme,sme-f8f16}}
    svmla_single_za16_mf8_vg2x4_fpm(slice, znx4, zn, fpmr);

    // expected-error@+1 {{'svmla_single_za32_mf8_vg4x1_fpm' needs target feature sme,sme-f8f32}}
    svmla_single_za32_mf8_vg4x1_fpm(slice, zn, zn, fpmr);

    // expected-error@+1 {{'svmla_single_za32_mf8_vg4x2_fpm' needs target feature sme,sme-f8f32}}
    svmla_single_za32_mf8_vg4x2_fpm(slice, znx2, zn, fpmr);

    // expected-error@+1 {{'svmla_single_za32_mf8_vg4x4_fpm' needs target feature sme,sme-f8f32}}
    svmla_single_za32_mf8_vg4x4_fpm(slice, znx4, zn, fpmr);

    // expected-error@+1 {{'svmla_za16_mf8_vg2x2_fpm' needs target feature sme,sme-f8f16}}
    svmla_za16_mf8_vg2x2_fpm(slice, znx2, znx2, fpmr);

    // expected-error@+1 {{'svmla_za16_mf8_vg2x4_fpm' needs target feature sme,sme-f8f16}}
    svmla_za16_mf8_vg2x4_fpm(slice, znx4, znx4, fpmr);

    // expected-error@+1 {{'svmla_za32_mf8_vg4x2_fpm' needs target feature sme,sme-f8f32}}
    svmla_za32_mf8_vg4x2_fpm(slice, znx2, znx2, fpmr);

    // expected-error@+1 {{'svmla_za32_mf8_vg4x4_fpm' needs target feature sme,sme-f8f32}}
    svmla_za32_mf8_vg4x4_fpm(slice, znx4, znx4, fpmr);
}