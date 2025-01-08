// RUN: %clang_cc1 -triple aarch64 -target-feature +sme -verify -emit-llvm-only %s

// REQUIRES: aarch64-registered-target

#include <arm_sme.h>

void test_features(svbool_t pn, svbool_t pm, svmfloat8_t zn, svmfloat8_t zm,
                   fpm_t fpmr) __arm_streaming __arm_inout("za") {
    // expected-error@+1 {{'svmopa_za16_mf8_m_fpm' needs target feature sme,sme-f8f16}}
    svmopa_za16_mf8_m_fpm(0, pn, pm, zn, zm, fpmr);
    // expected-error@+1 {{'svmopa_za32_mf8_m_fpm' needs target feature sme,sme-f8f32}}
    svmopa_za32_mf8_m_fpm(0, pn, pm, zn, zm, fpmr);
}
