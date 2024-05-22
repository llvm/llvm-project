// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "../cpu_model/aarch64.h"
#include <arm_sme.h>

__attribute__((visibility("hidden"), nocommon))
_Bool __aarch64_has_sme_and_tpidr2_el0;

// We have multiple ways to check that the function has SME, depending on our
// target.
// * For Linux we can use __getauxval().
// * For newlib we can use __aarch64_sme_accessible().

#if defined(__linux__)

#ifndef AT_HWCAP2
#define AT_HWCAP2 26
#endif

#ifndef HWCAP2_SME
#define HWCAP2_SME (1 << 23)
#endif

extern unsigned long int __getauxval (unsigned long int);

static _Bool has_sme(void) {
  return __getauxval(AT_HWCAP2) & HWCAP2_SME;
}

#else  // defined(__linux__)

#if defined(COMPILER_RT_SHARED_LIB)
__attribute__((weak))
#endif
extern _Bool __aarch64_sme_accessible(void);

static _Bool has_sme(void)  {
#if defined(COMPILER_RT_SHARED_LIB)
  if (!__aarch64_sme_accessible)
    return 0;
#endif
  return __aarch64_sme_accessible();
}

#endif // defined(__linux__)

#if __GNUC__ >= 9
#pragma GCC diagnostic ignored "-Wprio-ctor-dtor"
#endif
__attribute__((constructor(90)))
static void init_aarch64_has_sme(void) {
  __aarch64_has_sme_and_tpidr2_el0 = has_sme();
}

#if __GNUC__ >= 9
#pragma GCC diagnostic ignored "-Wprio-ctor-dtor"
#endif
__attribute__((constructor(90))) static void get_aarch64_cpu_features(void) {
  if (!get_features())
    __init_cpu_features();
}

extern bool __arm_in_streaming_mode(void) __arm_streaming_compatible;

__attribute__((target("sve"))) long
__arm_get_current_vg(void) __arm_streaming_compatible {
  bool HasSVE = get_features() & (1ULL << FEAT_SVE);
  if (!HasSVE && !has_sme())
    return 0;

  if (HasSVE || __arm_in_streaming_mode()) {
    long vl;
    __asm__ __volatile__("cntd %0" : "=r"(vl));
    return vl;
  }

  return 0;
}
