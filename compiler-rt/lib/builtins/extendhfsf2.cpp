//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements compiler-rt's __extendhfsf2, extend float16 to float,
/// on top of LLVM-libc's shared::extendhfsf2.
///
//===----------------------------------------------------------------------===//

#define SRC_HALF
#define DST_SINGLE
#include "fp_extend_impl.inc"

#include "fp_libc_config.h"
#include "shared/bit.h"
#include "shared/builtins/extendhfsf2.h"

extern "C" {
COMPILER_RT_ABI NOINLINE dst_t __extendhfsf2(src_t a) {
  return LIBC_NAMESPACE::shared::extendhfsf2(
      LIBC_NAMESPACE::shared::bit_cast<uint16_t>(a));
}

#if defined(__ARM_EABI__)
#if defined(COMPILER_RT_ARMHF_TARGET)
AEABI_RTABI dst_t __gnu_h2f_ieee(src_t a) { return __extendhfsf2(a); }
AEABI_RTABI dst_t __aeabi_h2f(src_t a) { return __extendhfsf2(a); }
#else
COMPILER_RT_ALIAS(__extendhfsf2, __gnu_h2f_ieee)
COMPILER_RT_ALIAS(__extendhfsf2, __aeabi_h2f)
#endif
#else
COMPILER_RT_ABI dst_t __gnu_h2f_ieee(src_t a) { return __extendhfsf2(a); }
#endif
}
