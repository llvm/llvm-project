//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements compiler-rt's __truncxfbf2, truncate long double to
/// bfloat16, on top of LLVM-libc's shared::truncxfbf2.
///
//===----------------------------------------------------------------------===//

#include "int_lib.h"

#if defined(CRT_HAS_TF_MODE) && __LDBL_MANT_DIG__ == 64 && defined(__x86_64__)
#define SRC_80
#define DST_BFLOAT
#include "fp_trunc.h"

#include "fp_libc_config.h"
#include "shared/builtins/truncxfbf2.h"

extern "C" COMPILER_RT_ABI dst_t __truncxfbf2(long double a) {
  return __builtin_bit_cast(dst_t, LIBC_NAMESPACE::shared::truncxfbf2(a));
}
#endif
