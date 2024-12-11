// RUN: %clang_cc1 -triple aarch64 -target-feature +sme -target-feature +sme2 -target-feature +sme-f8f16 -target-feature +sme-f8f32 -fsyntax-only -verify  %s

// REQUIRES: aarch64-registered-target

#include <arm_sme.h>

void test_svmopa(svbool_t pn, svbool_t pm, svmfloat8_t zn, svmfloat8_t zm,
                 fpm_t fpmr) __arm_streaming __arm_inout("za") {
    // expected-error@+1 {{argument value 18446744073709551615 is outside the valid range [0, 1]}}
    svmopa_za16_mf8_m_fpm(-1, pn, pm, zn, zm, fpmr);
    // expected-error@+1 {{argument value 2 is outside the valid range [0, 1]}}
    svmopa_za16_mf8_m_fpm(2, pn, pm, zn, zm, fpmr);

    // expected-error@+1 {{argument value 18446744073709551615 is outside the valid range [0, 3]}}
    svmopa_za32_mf8_m_fpm(-1, pn, pm, zn, zm, fpmr);
    // expected-error@+1 {{argument value 4 is outside the valid range [0, 3]}}
    svmopa_za32_mf8_m_fpm(4, pn, pm, zn, zm, fpmr);
}
