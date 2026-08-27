//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements compiler-rt's __truncdfhf2, truncate double to float16,
/// on top of LLVM-libc's shared::truncdfhf2.
///
//===----------------------------------------------------------------------===//

#define SRC_DOUBLE
#define DST_HALF
#include "fp_trunc_impl.inc"

#include "fp_libc_config.h"
#include "shared/bit.h"
#include "shared/builtins/truncdfhf2.h"

#ifdef LIBC_TYPES_HAS_FLOAT16
extern "C" COMPILER_RT_ABI dst_t __truncdfhf2(src_t a) {
  return LIBC_NAMESPACE::shared::bit_cast<dst_t>(
      LIBC_NAMESPACE::shared::truncdfhf2(a));
}
#else
extern "C" COMPILER_RT_ABI dst_t __truncdfhf2(src_t a) {
  return __truncXfYf2__(a);
}
#endif
