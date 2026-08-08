//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements compiler-rt's __truncsfhf2, truncate float to float16,
/// on top of LLVM-libc's shared::truncsfhf2.
///
//===----------------------------------------------------------------------===//

#define SRC_SINGLE
#define DST_HALF
#include "fp_trunc_impl.inc"

#include "fp_libc_config.h"
#include "shared/bit.h"
#include "shared/builtins/truncsfhf2.h"

#ifdef LIBC_TYPES_HAS_FLOAT16
extern "C" COMPILER_RT_ABI NOINLINE dst_t __truncsfhf2(src_t a) {
  return LIBC_NAMESPACE::shared::bit_cast<dst_t>(
      LIBC_NAMESPACE::shared::truncsfhf2(a));
}
#else
extern "C" COMPILER_RT_ABI NOINLINE dst_t __truncsfhf2(src_t a) {
  return __truncXfYf2__(a);
}
#endif

extern "C" {
#if defined(__ARM_EABI__)
#if defined(COMPILER_RT_ARMHF_TARGET)
AEABI_RTABI dst_t __gnu_f2h_ieee(float a) { return __truncsfhf2(a); }
AEABI_RTABI dst_t __aeabi_f2h(float a) { return __truncsfhf2(a); }
#else
COMPILER_RT_ALIAS(__truncsfhf2, __gnu_f2h_ieee)
COMPILER_RT_ALIAS(__truncsfhf2, __aeabi_f2h)
#endif
#else
COMPILER_RT_ABI dst_t __gnu_f2h_ieee(float a) { return __truncsfhf2(a); }
#endif
}
