// RUN: %clang_cc1 -triple aarch64 -target-feature +sme -verify -emit-llvm-only %s

// REQUIRES: aarch64-registered-target

#include <arm_sme.h>

void test_features(svbool_t pn, svbool_t pm,
                   svfloat16_t zn, svfloat16_t zm,
                   svbfloat16_t znb, svbfloat16_t zmb)
  __arm_streaming __arm_inout("za") {
// expected-error@+1 {{'svmopa_za16_bf16_m' needs target feature sme,sme-b16b16}}
  svmopa_za16_bf16_m(0, pn, pm, znb, zmb);
// expected-error@+1 {{'svmops_za16_bf16_m' needs target feature sme,sme-b16b16}}
  svmops_za16_bf16_m(0, pn, pm, znb, zmb);
// expected-error@+1 {{'svmopa_za16_f16_m' needs target feature sme,sme-f16f16}}
  svmopa_za16_f16_m(0, pn, pm, zn, zm);
// expected-error@+1 {{'svmops_za16_f16_m' needs target feature sme,sme-f16f16}}
  svmops_za16_f16_m(0, pn, pm, zn, zm);
}

void test_imm(svbool_t pn, svbool_t pm,
              svfloat16_t zn, svfloat16_t zm,
              svbfloat16_t znb, svbfloat16_t zmb)
  __arm_streaming __arm_inout("za") {
// expected-error@+1 {{argument value 18446744073709551615 is outside the valid range [0, 1]}}
  svmopa_za16_bf16_m(-1, pn, pm, znb, zmb);
// expected-error@+1 {{argument value 18446744073709551615 is outside the valid range [0, 1]}}
  svmops_za16_bf16_m(-1, pn, pm, znb, zmb);
// expected-error@+1 {{argument value 18446744073709551615 is outside the valid range [0, 1]}}
  svmopa_za16_f16_m(-1, pn, pm, zn, zm);
// expected-error@+1 {{argument value 18446744073709551615 is outside the valid range [0, 1]}}
  svmops_za16_f16_m(-1, pn, pm, zn, zm);
}

