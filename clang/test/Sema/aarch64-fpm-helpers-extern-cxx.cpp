// RUN: %clang_cc1 -triple aarch64-linux-gnu -fsyntax-only -verify -DUSE_NEON_H %s
// RUN: %clang_cc1 -triple aarch64-linux-gnu -fsyntax-only -verify -DUSE_SVE_H %s
// RUN: %clang_cc1 -triple aarch64-linux-gnu -fsyntax-only -verify -DUSE_SME_H %s
// RUN: %clang_cc1 -triple aarch64-linux-gnu -fsyntax-only -verify -DUSE_NEON_H -DUSE_SVE_H %s
// RUN: %clang_cc1 -triple aarch64_be-linux-gnu -fsyntax-only -verify -DUSE_NEON_H %s
// RUN: %clang_cc1 -triple aarch64-windows-msvc -fms-compatibility -fsyntax-only -verify -DUSE_NEON_H %s
// RUN: %clang_cc1 -triple aarch64-windows-msvc -fms-compatibility -fsyntax-only -verify -DUSE_SVE_H %s
// RUN: %clang_cc1 -triple aarch64-windows-msvc -fms-compatibility -fsyntax-only -verify -DUSE_SME_H %s
// RUN: %clang_cc1 -triple arm64ec-windows-msvc -fms-compatibility -fsyntax-only -verify -DUSE_NEON_H %s
// RUN: %clang_cc1 -triple arm64-apple-darwin -fsyntax-only -verify -DUSE_NEON_H %s

// REQUIRES: aarch64-registered-target

extern "C++" {
#ifdef USE_NEON_H
#include <arm_neon.h>
#endif

#ifdef USE_SVE_H
#include <arm_sve.h>
#endif

#ifdef USE_SME_H
#include <arm_sme.h>
#endif

template <typename T>
T check_cxx_linkage(T v) {
  return v;
}
}

fpm_t test_fpm_helpers(fpm_t fpm, uint64_t u, int64_t s) {
  fpm = check_cxx_linkage(__arm_fpm_init());
  fpm = __arm_set_fpm_lscale(fpm, 0);
  fpm = __arm_set_fpm_lscale(fpm, 127);
  fpm = __arm_set_fpm_lscale(fpm, u);

  fpm = __arm_set_fpm_nscale(fpm, -128);
  fpm = __arm_set_fpm_nscale(fpm, 127);
  fpm = __arm_set_fpm_nscale(fpm, s);

  fpm = __arm_set_fpm_lscale2(fpm, 0);
  fpm = __arm_set_fpm_lscale2(fpm, 63);
  fpm = __arm_set_fpm_lscale2(fpm, u);

  __arm_set_fpm_lscale(fpm, 128);
  // expected-error@-1 {{argument value 128 is outside the valid range [0, 127]}}

  __arm_set_fpm_nscale(fpm, -129);
  // expected-error@-1 {{argument value -129 is outside the valid range [-128, 127]}}

  __arm_set_fpm_lscale2(fpm, 64);
  // expected-error@-1 {{argument value 64 is outside the valid range [0, 63]}}

  return fpm;
}
