// RUN: %clang_cc1 -triple thumbv8.1m.main-arm-none-eabi -fallow-half-arguments-and-returns -target-feature +mve.fp -target-feature +cdecp0 -verify -fsyntax-only %s

#include <arm_cde.h>
#include <arm_acle.h>

void test_coproc_gcp_instr(int a) {
  __builtin_arm_cdp(0, 2, 3, 4, 5, 6); // expected-error {{coprocessor 0 must be configured as GCP}}
  __builtin_arm_cdp2(0, 2, 3, 4, 5, 6); // expected-error {{coprocessor 0 must be configured as GCP}}
  __builtin_arm_mcr(0, 0, a, 13, 0, 3); // expected-error {{coprocessor 0 must be configured as GCP}}
  __builtin_arm_mcr2(0, 0, a, 13, 0, 3); // expected-error {{coprocessor 0 must be configured as GCP}}
  __builtin_arm_mrc(0, 0, 13, 0, 3); // expected-error {{coprocessor 0 must be configured as GCP}}
  __builtin_arm_mrc2(0, 0, 13, 0, 3); // expected-error {{coprocessor 0 must be configured as GCP}}
  __builtin_arm_mcrr(0, 0, a, 0); // expected-error {{coprocessor 0 must be configured as GCP}}
  __builtin_arm_mcrr2(0, 0, a, 0); // expected-error {{coprocessor 0 must be configured as GCP}}
  __builtin_arm_mrrc(0, 0, 0); // expected-error {{coprocessor 0 must be configured as GCP}}
  __builtin_arm_mrrc2(0, 0, 0); // expected-error {{coprocessor 0 must be configured as GCP}}
  __builtin_arm_ldc(0, 2, &a); // expected-error {{coprocessor 0 must be configured as GCP}}
  __builtin_arm_ldcl(0, 2, &a); // expected-error {{coprocessor 0 must be configured as GCP}}
  __builtin_arm_ldc2(0, 2, &a); // expected-error {{coprocessor 0 must be configured as GCP}}
  __builtin_arm_ldc2l(0, 2, &a); // expected-error {{coprocessor 0 must be configured as GCP}}
  __builtin_arm_stc(0, 2, &a); // expected-error {{coprocessor 0 must be configured as GCP}}
  __builtin_arm_stcl(0, 2, &a); // expected-error {{coprocessor 0 must be configured as GCP}}
  __builtin_arm_stc2(0, 2, &a); // expected-error {{coprocessor 0 must be configured as GCP}}
  __builtin_arm_stc2l(0, 2, &a); // expected-error {{coprocessor 0 must be configured as GCP}}
}

void test_coproc(uint32_t a) {
  (void)__arm_cx1(0, 0);
  __arm_cx1(a, 0); // expected-error {{argument to '__arm_cx1' must be a constant integer}}
  __arm_cx1(-1, 0); // expected-error {{argument value -1 is outside the valid range [0, 7]}}
  __arm_cx1(8, 0);  // expected-error {{argument value 8 is outside the valid range [0, 7]}}
  __arm_cx1(1, 0); // expected-error {{coprocessor 1 must be configured as CDE}}
}

void test_cx(uint32_t a) {
  (void)__arm_cx1(0, 0);
  __arm_cx1(a, 0); // expected-error {{argument to '__arm_cx1' must be a constant integer}}
  __arm_cx1(0, a);  // expected-error {{argument to '__arm_cx1' must be a constant integer}}
  __arm_cx1(0, 8192);  // expected-error {{argument value 8192 is outside the valid range [0, 8191]}}
}
