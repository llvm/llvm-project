// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "../cpu_model/aarch64.h"

struct FEATURES {
  unsigned long long features;
};

extern struct FEATURES __aarch64_cpu_features;

struct SME_STATE {
  long PSTATE;
  long TPIDR2_EL0;
};

extern struct SME_STATE __arm_sme_state(void) __arm_streaming_compatible;

extern bool __aarch64_has_sme_and_tpidr2_el0;

#if __GNUC__ >= 9
#pragma GCC diagnostic ignored "-Wprio-ctor-dtor"
#endif
__attribute__((constructor(90))) static void get_aarch64_cpu_features(void) {
  if (__atomic_load_n(&__aarch64_cpu_features.features, __ATOMIC_RELAXED))
    return;

  __init_cpu_features();
}

__attribute__((target("sve"))) long
__arm_get_current_vg(void) __arm_streaming_compatible {
  struct SME_STATE State = __arm_sme_state();
  unsigned long long features =
      __atomic_load_n(&__aarch64_cpu_features.features, __ATOMIC_RELAXED);
  bool HasSVE = features & (1ULL << FEAT_SVE);

  if (!HasSVE && !__aarch64_has_sme_and_tpidr2_el0)
    return 0;

  if (HasSVE || (State.PSTATE & 1)) {
    long vl;
    __asm__ __volatile__("cntd %0" : "=r"(vl));
    return vl;
  }

  return 0;
}
