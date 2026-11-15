// RUN: %clang_cc1 -triple aarch64 -fsyntax-only -verify -DUSE_NEON_H %s
// RUN: %clang_cc1 -triple aarch64 -fsyntax-only -verify -DUSE_SVE_H %s
// RUN: %clang_cc1 -triple aarch64 -fsyntax-only -verify -DUSE_SME_H %s
// RUN: %clang_cc1 -triple aarch64 -x c++ -fsyntax-only -verify -DUSE_NEON_H %s
// RUN: %clang_cc1 -triple aarch64 -x c++ -fsyntax-only -verify -DUSE_SVE_H %s
// RUN: %clang_cc1 -triple aarch64 -x c++ -fsyntax-only -verify -DUSE_SME_H %s

// REQUIRES: aarch64-registered-target

#ifdef USE_NEON_H
#include "arm_neon.h"
#endif

#ifdef USE_SVE_H
#include "arm_sve.h"
#endif

#ifdef USE_SME_H
#include "arm_sme.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif

void test_lscale(fpm_t fpm, uint64_t variable) {
  __arm_set_fpm_lscale(fpm, 0);
  __arm_set_fpm_lscale(fpm, 127);
  __arm_set_fpm_lscale(fpm, variable);

  __arm_set_fpm_lscale(fpm, 128);
  // expected-error@-1 {{argument value 128 is outside the valid range [0, 127]}}

  __arm_set_fpm_lscale(fpm, -1);
  // expected-error@-1 {{argument value 18446744073709551615 is outside the valid range [0, 127]}}
}

void test_nscale(fpm_t fpm, int64_t variable) {
  __arm_set_fpm_nscale(fpm, -128);
  __arm_set_fpm_nscale(fpm, 127);
  __arm_set_fpm_nscale(fpm, variable);

  __arm_set_fpm_nscale(fpm, -129);
  // expected-error@-1 {{argument value -129 is outside the valid range [-128, 127]}}

  __arm_set_fpm_nscale(fpm, 128);
  // expected-error@-1 {{argument value 128 is outside the valid range [-128, 127]}}
}

void test_lscale2(fpm_t fpm, uint64_t variable) {
  __arm_set_fpm_lscale2(fpm, 0);
  __arm_set_fpm_lscale2(fpm, 63);
  __arm_set_fpm_lscale2(fpm, variable);

  __arm_set_fpm_lscale2(fpm, 64);
  // expected-error@-1 {{argument value 64 is outside the valid range [0, 63]}}

  __arm_set_fpm_lscale2(fpm, -1);
  // expected-error@-1 {{argument value 18446744073709551615 is outside the valid range [0, 63]}}
}
#ifdef __cplusplus
}
#endif
