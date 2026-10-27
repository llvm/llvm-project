// RUN: %clang_cc1 -triple aarch64-linux-gnu -ffreestanding -fsyntax-only -verify %s
// RUN: %clang_cc1 -triple aarch64-linux-gnu -ffreestanding -fsyntax-only -verify -x c++ %s
// RUN: %clang_cc1 -triple arm64-apple-darwin -ffreestanding -fsyntax-only -verify %s
// RUN: %clang_cc1 -triple aarch64-windows-msvc -ffreestanding -fsyntax-only -verify %s

// REQUIRES: aarch64-registered-target

// Verify that the declarations of the FPM scale helpers in arm_vector_types.h
// match the builtin prototypes even if the 64-bit integer types have been
// redefined, as the Linux kernel does in arch/arm64/include/asm/neon-intrinsics.h.

// expected-no-diagnostics

#undef __INT64_TYPE__
#define __INT64_TYPE__ long long
#undef __UINT64_TYPE__
#define __UINT64_TYPE__ unsigned long long

#include <arm_neon.h>

#ifdef __cplusplus
extern "C" {
#endif

fpm_t test(fpm_t fpm, uint64_t uscale, int64_t sscale) {
  fpm = __arm_set_fpm_lscale(fpm, uscale);
  fpm = __arm_set_fpm_nscale(fpm, sscale);
  return __arm_set_fpm_lscale2(fpm, 63);
}

#ifdef __cplusplus
}
#endif
