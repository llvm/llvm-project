// RUN: %clang_cc1 -triple aarch64 \
// RUN:   -target-feature +sme -target-feature +sme2 -verify -emit-llvm -o - %s

// REQUIRES: aarch64-registered-target

#include <arm_sme.h>

void test_features(svuint8x2_t zn_u8,      svuint8_t zm_u8,
                   svint8x2_t zn_s8,       svint8_t zm_s8,
                   svint16x2_t zn_s16,     svint16_t zm_s16,
                   svuint16x2_t zn_u16,    svuint16_t zm_u16,
                   svfloat16x2_t zn_f16,   svfloat16_t zm_f16,
                   svbfloat16x2_t zn_bf16, svbfloat16_t zm_bf16,
                   svfloat32x2_t zn_f32,   svfloat32_t zm_f32,
                   svmfloat8x2_t zn_f8,    svmfloat8_t zm_f8,
                   svuint8_t zk,           fpm_t fpm) __arm_streaming __arm_inout("za") {

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
// expected-error@+1 {{'svtmopa_lane_za32_f32_f32' needs target feature sme,sme2,sme-tmop}}
    svtmopa_lane_za32_f32_f32(0, zn_f32, zm_f32, zk, 0);
// expected-error@+1 {{'svtmopa_lane_za32_bf16_bf16' needs target feature sme,sme2,sme-tmop}}
    svtmopa_lane_za32_bf16_bf16(0, zn_bf16, zm_bf16, zk, 0);
// expected-error@+1 {{'svtmopa_lane_za16_f16_f16' needs target feature sme,sme2,sme-tmop,sme-f16f16}}
    svtmopa_lane_za16_f16_f16(0, zn_f16, zm_f16, zk, 0);
// expected-error@+1 {{'svtmopa_lane_za16_bf16_bf16' needs target feature sme,sme2,sme-tmop,sme-b16b16}}
    svtmopa_lane_za16_bf16_bf16(0, zn_bf16, zm_bf16, zk, 0);
// expected-error@+1 {{'svtmopa_lane_za16_mf8_mf8_fpm' needs target feature sme,sme2,sme-tmop,sme-f8f16}}
    svtmopa_lane_za16_mf8_mf8_fpm(0, zn_f8, zm_f8, zk, 0, fpm);
// expected-error@+1 {{'svtmopa_lane_za32_mf8_mf8_fpm' needs target feature sme,sme2,sme-tmop,sme-f8f32}}
    svtmopa_lane_za32_mf8_mf8_fpm(0, zn_f8, zm_f8, zk, 0, fpm);
}

void test_imm(svuint8x2_t zn_u8,      svuint8_t zm_u8,
              svint8x2_t zn_s8,       svint8_t zm_s8,
              svint16x2_t zn_s16,     svint16_t zm_s16,
              svuint16x2_t zn_u16,    svuint16_t zm_u16,
              svfloat16x2_t zn_f16,   svfloat16_t zm_f16,
              svbfloat16x2_t zn_bf16, svbfloat16_t zm_bf16,
              svfloat32x2_t zn_f32,   svfloat32_t zm_f32,
              svmfloat8x2_t zn_f8,    svmfloat8_t zm_f8,
              svuint8_t zk,           fpm_t fpm) __arm_streaming __arm_inout("za") {

// expected-error@+1 {{argument value 4 is outside the valid range [0, 3]}}
    svtmopa_lane_za32_s8_s8(0, zn_s8, zm_s8, zk, 4);
// expected-error@+1 {{argument value 4 is outside the valid range [0, 3]}}
    svtmopa_lane_za32_s8_s8(4, zn_s8, zm_s8, zk, 0);

// expected-error@+1 {{argument value 4 is outside the valid range [0, 3]}}
    svtmopa_lane_za32_u8_u8(0, zn_u8, zm_u8, zk, 4);
// expected-error@+1 {{argument value 4 is outside the valid range [0, 3]}}
    svtmopa_lane_za32_u8_u8(4, zn_u8, zm_u8, zk, 0);

// expected-error@+1 {{argument value 4 is outside the valid range [0, 3]}}
    svtmopa_lane_za32_s8_u8(0, zn_s8, zm_u8, zk, 4);
// expected-error@+1 {{argument value 4 is outside the valid range [0, 3]}}
    svtmopa_lane_za32_s8_u8(4, zn_s8, zm_u8, zk, 0);

// expected-error@+1 {{argument value 4 is outside the valid range [0, 3]}}
    svtmopa_lane_za32_u8_s8(0, zn_u8, zm_s8, zk, 4);
// expected-error@+1 {{argument value 4 is outside the valid range [0, 3]}}
    svtmopa_lane_za32_u8_s8(4, zn_u8, zm_s8, zk, 0);

// expected-error@+1 {{argument value 4 is outside the valid range [0, 3]}}
    svtmopa_lane_za32_s16_s16(0, zn_s16, zm_s16, zk, 4);
// expected-error@+1 {{argument value 4 is outside the valid range [0, 3]}}
    svtmopa_lane_za32_s16_s16(4, zn_s16, zm_s16, zk, 0);

// expected-error@+1 {{argument value 4 is outside the valid range [0, 3]}}
    svtmopa_lane_za32_u16_u16(0, zn_u16, zm_u16, zk, 4);
// expected-error@+1 {{argument value 4 is outside the valid range [0, 3]}}
    svtmopa_lane_za32_u16_u16(4, zn_u16, zm_u16, zk, 0);

// expected-error@+1 {{argument value 4 is outside the valid range [0, 3]}}
    svtmopa_lane_za32_f16_f16(0, zn_f16, zm_f16, zk, 4);
// expected-error@+1 {{argument value 4 is outside the valid range [0, 3]}}
    svtmopa_lane_za32_f16_f16(4, zn_f16, zm_f16, zk, 0);

// expected-error@+1 {{argument value 4 is outside the valid range [0, 3]}}
    svtmopa_lane_za32_f32_f32(0, zn_f32, zm_f32, zk, 4);
// expected-error@+1 {{argument value 4 is outside the valid range [0, 3]}}
    svtmopa_lane_za32_f32_f32(4, zn_f32, zm_f32, zk, 0);

// expected-error@+1 {{argument value 4 is outside the valid range [0, 3]}}
    svtmopa_lane_za32_bf16_bf16(0, zn_bf16, zm_bf16, zk, 4);
// expected-error@+1 {{argument value 4 is outside the valid range [0, 3]}}
    svtmopa_lane_za32_bf16_bf16(4, zn_bf16, zm_bf16, zk, 0);

// expected-error@+1 {{argument value 4 is outside the valid range [0, 3]}}
    svtmopa_lane_za16_f16_f16(0, zn_f16, zm_f16, zk, 4);
// expected-error@+1 {{argument value 2 is outside the valid range [0, 1]}}
    svtmopa_lane_za16_f16_f16(2, zn_f16, zm_f16, zk, 0);

// expected-error@+1 {{argument value 4 is outside the valid range [0, 3]}}
    svtmopa_lane_za16_bf16_bf16(0, zn_bf16, zm_bf16, zk, 4);
// expected-error@+1 {{argument value 2 is outside the valid range [0, 1]}}
    svtmopa_lane_za16_bf16_bf16(2, zn_bf16, zm_bf16, zk, 0);

// expected-error@+1 {{argument value 4 is outside the valid range [0, 3]}}
    svtmopa_lane_za16_mf8_mf8_fpm(0, zn_f8, zm_f8, zk, 4, fpm);
// expected-error@+1 {{argument value 2 is outside the valid range [0, 1]}}
    svtmopa_lane_za16_mf8_mf8_fpm(2, zn_f8, zm_f8, zk, 0, fpm);

// expected-error@+1 {{argument value 4 is outside the valid range [0, 3]}}
    svtmopa_lane_za32_mf8_mf8_fpm(0, zn_f8, zm_f8, zk, 4, fpm);
// expected-error@+1 {{argument value 4 is outside the valid range [0, 3]}}
    svtmopa_lane_za32_mf8_mf8_fpm(4, zn_f8, zm_f8, zk, 0, fpm);
}
