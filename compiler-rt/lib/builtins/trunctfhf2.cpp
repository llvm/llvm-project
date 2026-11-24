//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements compiler-rt's __trunctfhf2, truncate float128 to
/// float16, on top of LLVM-libc's shared::trunctfhf2.
///
//===----------------------------------------------------------------------===//

#define QUAD_PRECISION
#include "fp_lib.h"

#include "fp_libc_config.h"
#include "shared/bit.h"
#include "shared/builtins/trunctfhf2.h"

#ifdef CRT_HAS_TF_MODE
#define SRC_QUAD
#define DST_HALF
#include "fp_trunc_impl.inc"

extern "C" COMPILER_RT_ABI dst_t __trunctfhf2(src_t a) {
  return LIBC_NAMESPACE::shared::bit_cast<dst_t>(
      LIBC_NAMESPACE::shared::trunctfhf2(a));
}
#endif
