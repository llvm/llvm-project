// RUN: %clang_cc1 -triple aarch64 -target-feature +sme -target-feature +sme2 -target-feature +bf16 -target-feature +sme-f16f16 -target-feature +sme-b16b16 -verify -emit-llvm -o - %s

// REQUIRES: aarch64-registered-target

#include <arm_sme.h>

void test_features() __arm_streaming __arm_inout("za") {
    svuint8x2_t zn_u8;
    svint8x2_t zn_s8;
    svuint8_t zm_u8;
    svint8_t zm_s8;
    svuint16x2_t zn_u16;
    svint16x2_t zn_s16;
    svuint16_t zm_u16;
    svint16_t zm_s16;
    svbfloat16x2_t zn_bf16;
    svfloat16x2_t zn_f16;
    svbfloat16_t zm_bf16;
    svfloat16_t zm_f16;
    svfloat32x2_t zn_f32;
    svfloat32_t zm_f32;
    fpm_t fpm = 0;
    svuint8_t zk;

// expected-error@+1 {{'svtmopa_lane_za32_s8_s8' needs target feature sme,sme2,sme-tmop}}
    svtmopa_lane_za32_s8_s8(0, zn_s8, zm_s8, zk, 0);
// expected-error@+1 {{'svtmopa_lane_za32_u8_u8' needs target feature sme,sme2,sme-tmop}}
    svtmopa_lane_za32_u8_u8(0, zn_u8, zm_u8, zk, 0);
// expected-error@+1 {{'svtmopa_lane_za32_s8_u8' needs target feature sme,sme2,sme-tmop}}
    svtmopa_lane_za32_s8_u8(0, zn_s8, zm_u8, zk, 0);
// expected-error@+1 {{'svtmopa_lane_za32_u8_s8' needs target feature sme,sme2,sme-tmop}}
    svtmopa_lane_za32_u8_s8(0, zn_u8, zm_s8, zk, 0);
// expected-error@+1 {{'svtmopa_lane_za32_s16_s16' needs target feature sme,sme2,sme-tmop}}
    svtmopa_lane_za32_s16_s16(0, zn_s16, zm_s16, zk, 0);
// expected-error@+1 {{'svtmopa_lane_za32_u16_u16' needs target feature sme,sme2,sme-tmop}}
    svtmopa_lane_za32_u16_u16(0, zn_u16, zm_u16, zk, 0);
// expected-error@+1 {{'svtmopa_lane_za32_f16_f16' needs target feature sme,sme2,sme-tmop}}
    svtmopa_lane_za32_f16_f16(0, zn_f16, zm_f16, zk, 0);
// expected-error@+1 {{'svtmopa_lane_za32_bf16_bf16' needs target feature sme,sme2,sme-tmop}}
    svtmopa_lane_za32_bf16_bf16(0, zn_bf16, zm_bf16, zk, 0);
// expected-error@+1 {{'svtmopa_lane_za16_f16_f16' needs target feature sme,sme2,sme-tmop,sme-f16f16}}
    svtmopa_lane_za16_f16_f16(0, zn_f16, zm_f16, zk, 0);
// expected-error@+1 {{'svtmopa_lane_za16_bf16_bf16' needs target feature sme,sme2,sme-tmop,sme-f16f16}}
    svtmopa_lane_za16_bf16_bf16(0, zn_bf16, zm_bf16, zk, 0);
// expected-error@+1 {{'svtmopa_lane_za16_mf8_mf8_fpm' needs target feature sme,sme2,sme-tmop,sme-f8f16}}
    svtmopa_lane_za16_mf8_mf8_fpm(0, zn_f32, zm_f32, zk, 0, fpm);
// expected-error@+1 {{'svtmopa_lane_za32_mf8_mf8_fpm' needs target feature sme,sme2,sme-tmop,sme-f8f32}}
    svtmopa_lane_za32_mf8_mf8_fpm(0, zn_f32, zm_f32, zk, 0, fpm);
}

void test_imm() __arm_streaming __arm_inout("za") {
    svuint8x2_t zn_u8;
    svint8x2_t zn_s8;
    svuint8_t zm_u8;
    svint8_t zm_s8;
    svuint16x2_t zn_u16;
    svint16x2_t zn_s16;
    svuint16_t zm_u16;
    svint16_t zm_s16;
    svbfloat16x2_t zn_bf16;
    svfloat16x2_t zn_f16;
    svbfloat16_t zm_bf16;
    svfloat16_t zm_f16;
    svfloat32x2_t zn_f32;
    svfloat32_t zm_f32;
    fpm_t fpm;
    svuint8_t zk;

// expected-error@+1 {{argument value 4 is outside the valid range [0, 3]}}
    svtmopa_lane_za32_s8_s8(3, zn_s8, zm_s8, zk, 4);
// expected-error@+1 {{argument value 4 is outside the valid range [0, 3]}}
    svtmopa_lane_za32_s8_s8(4, zn_s8, zm_s8, zk, 3);
// expected-error@+1 {{argument value 18446744073709551615 is outside the valid range [0, 3]}}
    svtmopa_lane_za32_s8_s8(0, zn_s8, zm_s8, zk, -1);
// expected-error@+1 {{argument value 18446744073709551615 is outside the valid range [0, 3]}}
    svtmopa_lane_za32_s8_s8(-1, zn_s8, zm_s8, zk, 0);

// expected-error@+1 {{argument value 4 is outside the valid range [0, 3]}}
    svtmopa_lane_za32_u8_u8(3, zn_u8, zm_u8, zk, 4);
// expected-error@+1 {{argument value 4 is outside the valid range [0, 3]}}
    svtmopa_lane_za32_u8_u8(4, zn_u8, zm_u8, zk, 3);
// expected-error@+1 {{argument value 18446744073709551615 is outside the valid range [0, 3]}}
    svtmopa_lane_za32_u8_u8(0, zn_u8, zm_u8, zk, -1);
// expected-error@+1 {{argument value 18446744073709551615 is outside the valid range [0, 3]}}
    svtmopa_lane_za32_u8_u8(-1, zn_u8, zm_u8, zk, 0);

// expected-error@+1 {{argument value 4 is outside the valid range [0, 3]}}
    svtmopa_lane_za32_s8_u8(3, zn_s8, zm_u8, zk, 4);
// expected-error@+1 {{argument value 4 is outside the valid range [0, 3]}}
    svtmopa_lane_za32_s8_u8(4, zn_s8, zm_u8, zk, 3);
// expected-error@+1 {{argument value 18446744073709551615 is outside the valid range [0, 3]}}
    svtmopa_lane_za32_s8_u8(0, zn_s8, zm_u8, zk, -1);
// expected-error@+1 {{argument value 18446744073709551615 is outside the valid range [0, 3]}}
    svtmopa_lane_za32_s8_u8(-1, zn_s8, zm_u8, zk, 0);

// expected-error@+1 {{argument value 4 is outside the valid range [0, 3]}}
    svtmopa_lane_za32_u8_s8(3, zn_u8, zm_s8, zk, 4);
// expected-error@+1 {{argument value 4 is outside the valid range [0, 3]}}
    svtmopa_lane_za32_u8_s8(4, zn_u8, zm_s8, zk, 3);
// expected-error@+1 {{argument value 18446744073709551615 is outside the valid range [0, 3]}}
    svtmopa_lane_za32_u8_s8(0, zn_u8, zm_s8, zk, -1);
// expected-error@+1 {{argument value 18446744073709551615 is outside the valid range [0, 3]}}
    svtmopa_lane_za32_u8_s8(-1, zn_u8, zm_s8, zk, 0);

// expected-error@+1 {{argument value 4 is outside the valid range [0, 3]}}
    svtmopa_lane_za32_s16_s16(3, zn_s16, zm_s16, zk, 4);
// expected-error@+1 {{argument value 4 is outside the valid range [0, 3]}}
    svtmopa_lane_za32_s16_s16(4, zn_s16, zm_s16, zk, 3);
// expected-error@+1 {{argument value 18446744073709551615 is outside the valid range [0, 3]}}
    svtmopa_lane_za32_s16_s16(0, zn_s16, zm_s16, zk, -1);
// expected-error@+1 {{argument value 18446744073709551615 is outside the valid range [0, 3]}}
    svtmopa_lane_za32_s16_s16(-1, zn_s16, zm_s16, zk, 0);

// expected-error@+1 {{argument value 4 is outside the valid range [0, 3]}}
    svtmopa_lane_za32_u16_u16(3, zn_u16, zm_u16, zk, 4);
// expected-error@+1 {{argument value 4 is outside the valid range [0, 3]}}
    svtmopa_lane_za32_u16_u16(4, zn_u16, zm_u16, zk, 3);
// expected-error@+1 {{argument value 18446744073709551615 is outside the valid range [0, 3]}}
    svtmopa_lane_za32_u16_u16(0, zn_u16, zm_u16, zk, -1);
// expected-error@+1 {{argument value 18446744073709551615 is outside the valid range [0, 3]}}
    svtmopa_lane_za32_u16_u16(-1, zn_u16, zm_u16, zk, 0);

// expected-error@+1 {{argument value 4 is outside the valid range [0, 3]}}
    svtmopa_lane_za32_f16_f16(3, zn_f16, zm_f16, zk, 4);
// expected-error@+1 {{argument value 4 is outside the valid range [0, 3]}}
    svtmopa_lane_za32_f16_f16(4, zn_f16, zm_f16, zk, 3);
// expected-error@+1 {{argument value 18446744073709551615 is outside the valid range [0, 3]}}
    svtmopa_lane_za32_f16_f16(0, zn_f16, zm_f16, zk, -1);
// expected-error@+1 {{argument value 18446744073709551615 is outside the valid range [0, 3]}}
    svtmopa_lane_za32_f16_f16(-1, zn_f16, zm_f16, zk, 0);

// expected-error@+1 {{argument value 4 is outside the valid range [0, 3]}}
    svtmopa_lane_za32_bf16_bf16(3, zn_bf16, zm_bf16, zk, 4);
// expected-error@+1 {{argument value 4 is outside the valid range [0, 3]}}
    svtmopa_lane_za32_bf16_bf16(4, zn_bf16, zm_bf16, zk, 3);
// expected-error@+1 {{argument value 18446744073709551615 is outside the valid range [0, 3]}}
    svtmopa_lane_za32_bf16_bf16(0, zn_bf16, zm_bf16, zk, -1);
// expected-error@+1 {{argument value 18446744073709551615 is outside the valid range [0, 3]}}
    svtmopa_lane_za32_bf16_bf16(-1, zn_bf16, zm_bf16, zk, 0);

// expected-error@+1 {{argument value 4 is outside the valid range [0, 3]}}
    svtmopa_lane_za16_f16_f16(3, zn_f16, zm_f16, zk, 4);
// expected-error@+1 {{argument value 4 is outside the valid range [0, 3]}}
    svtmopa_lane_za16_f16_f16(4, zn_f16, zm_f16, zk, 3);
// expected-error@+1 {{argument value 18446744073709551615 is outside the valid range [0, 3]}}
    svtmopa_lane_za16_f16_f16(0, zn_f16, zm_f16, zk, -1);
// expected-error@+1 {{argument value 18446744073709551615 is outside the valid range [0, 3]}}
    svtmopa_lane_za16_f16_f16(-1, zn_f16, zm_f16, zk, 0);

// expected-error@+1 {{argument value 4 is outside the valid range [0, 3]}}
    svtmopa_lane_za16_bf16_bf16(3, zn_bf16, zm_bf16, zk, 4);
// expected-error@+1 {{argument value 4 is outside the valid range [0, 3]}}
    svtmopa_lane_za16_bf16_bf16(4, zn_bf16, zm_bf16, zk, 3);
// expected-error@+1 {{argument value 18446744073709551615 is outside the valid range [0, 3]}}
    svtmopa_lane_za16_bf16_bf16(0, zn_bf16, zm_bf16, zk, -1);
// expected-error@+1 {{argument value 18446744073709551615 is outside the valid range [0, 3]}}
    svtmopa_lane_za16_bf16_bf16(-1, zn_bf16, zm_bf16, zk, 0);

// expected-error@+1 {{argument value 4 is outside the valid range [0, 3]}}
    svtmopa_lane_za16_mf8_mf8_fpm(3, zn_f32, zm_f32, zk, 4, fpm);
// expected-error@+1 {{argument value 4 is outside the valid range [0, 3]}}
    svtmopa_lane_za16_mf8_mf8_fpm(4, zn_f32, zm_f32, zk, 3, fpm);
// expected-error@+1 {{argument value 18446744073709551615 is outside the valid range [0, 3]}}
    svtmopa_lane_za16_mf8_mf8_fpm(0, zn_f32, zm_f32, zk, -1, fpm);
// expected-error@+1 {{argument value 18446744073709551615 is outside the valid range [0, 3]}}
    svtmopa_lane_za16_mf8_mf8_fpm(-1, zn_f32, zm_f32, zk, 0, fpm);

// expected-error@+1 {{argument value 4 is outside the valid range [0, 3]}}
    svtmopa_lane_za32_mf8_mf8_fpm(3, zn_f32, zm_f32, zk, 4, fpm);
// expected-error@+1 {{argument value 4 is outside the valid range [0, 3]}}
    svtmopa_lane_za32_mf8_mf8_fpm(4, zn_f32, zm_f32, zk, 3, fpm);
// expected-error@+1 {{argument value 18446744073709551615 is outside the valid range [0, 3]}}
    svtmopa_lane_za32_mf8_mf8_fpm(0, zn_f32, zm_f32, zk, -1, fpm);
// expected-error@+1 {{argument value 18446744073709551615 is outside the valid range [0, 3]}}
    svtmopa_lane_za32_mf8_mf8_fpm(-1, zn_f32, zm_f32, zk, 0, fpm);
}
