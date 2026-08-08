//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements compiler-rt's __fixunsxfti, truncating long double ->
/// __uint128_t conversion (saturating), on top of LLVM-libc's
/// shared::fixunsxfti.
///
//===----------------------------------------------------------------------===//

#include "int_lib.h"

#include "fp_libc_config.h"
#include "shared/builtins/fixunsxfti.h"

#if defined(CRT_HAS_128BIT)
extern "C" COMPILER_RT_ABI tu_int __fixunsxfti(xf_float a) {
  return LIBC_NAMESPACE::shared::fixunsxfti(a);
}
#endif
