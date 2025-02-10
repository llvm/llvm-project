// RUN: %clang_cc1 -triple aarch64 -target-feature +sme -verify -emit-llvm-only %s

// REQUIRES: aarch64-registered-target

#include <arm_sve.h>


void test_features_sme2_fp8(svmfloat8_t zn, svfloat16x2_t znf16, svbfloat16x2_t znbf16,
                            svfloat32x4_t znf32,  fpm_t fpmr) __arm_streaming {
    // expected-error@+1 {{'svcvtl1_f16_mf8_x2_fpm' needs target feature sme,sme2,fp8}}
    svcvtl1_f16_mf8_x2_fpm(zn, fpmr);
    // expected-error@+1 {{'svcvtl2_f16_mf8_x2_fpm' needs target feature sme,sme2,fp8}}
    svcvtl2_f16_mf8_x2_fpm(zn, fpmr);
    // expected-error@+1 {{'svcvtl1_bf16_mf8_x2_fpm' needs target feature sme,sme2,fp8}}
    svcvtl1_bf16_mf8_x2_fpm(zn, fpmr);
    // expected-error@+1 {{'svcvtl2_bf16_mf8_x2_fpm' needs target feature sme,sme2,fp8}}
    svcvtl2_bf16_mf8_x2_fpm(zn, fpmr);

    // expected-error@+1 {{'svcvt1_f16_mf8_x2_fpm' needs target feature sme,sme2,fp8}}
    svcvt1_f16_mf8_x2_fpm(zn, fpmr);
    // expected-error@+1 {{'svcvt2_f16_mf8_x2_fpm' needs target feature sme,sme2,fp8}}
    svcvt2_f16_mf8_x2_fpm(zn, fpmr);
    // expected-error@+1 {{'svcvt1_bf16_mf8_x2_fpm' needs target feature sme,sme2,fp8}}
    svcvt1_bf16_mf8_x2_fpm(zn, fpmr);
    // expected-error@+1 {{'svcvt2_bf16_mf8_x2_fpm' needs target feature sme,sme2,fp8}}
    svcvt2_bf16_mf8_x2_fpm(zn, fpmr);

    // expected-error@+1 {{'svcvt_mf8_f16_x2_fpm' needs target feature sme,sme2,fp8}}
    svcvt_mf8_f16_x2_fpm(znf16, fpmr);
    // expected-error@+1 {{'svcvt_mf8_bf16_x2_fpm' needs target feature sme,sme2,fp8}}
    svcvt_mf8_bf16_x2_fpm(znbf16, fpmr);
    // expected-error@+1 {{'svcvt_mf8_f32_x4_fpm' needs target feature sme,sme2,fp8}}
    svcvt_mf8_f32_x4_fpm(znf32, fpmr);
    // expected-error@+1 {{'svcvtn_mf8_f32_x4_fpm' needs target feature sme,sme2,fp8}}
    svcvtn_mf8_f32_x4_fpm(znf32, fpmr);
}