//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements compiler-rt's __trunctfbf2, truncate float128 to
/// bfloat16, on top of LLVM-libc's shared::trunctfbf2.
///
//===----------------------------------------------------------------------===//

#define QUAD_PRECISION
#include "fp_lib.h"

#include "fp_libc_config.h"
#include "shared/builtins/trunctfbf2.h"

#if defined(CRT_HAS_TF_MODE) && defined(__x86_64__)
#define SRC_QUAD
#define DST_BFLOAT
#include "fp_trunc.h"

extern "C" COMPILER_RT_ABI dst_t __trunctfbf2(src_t a) {
  return __builtin_bit_cast(dst_t, LIBC_NAMESPACE::shared::trunctfbf2(a));
}
#endif
