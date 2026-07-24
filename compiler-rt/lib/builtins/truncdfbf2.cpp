//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements compiler-rt's __truncdfbf2, truncate double to
/// bfloat16, on top of LLVM-libc's shared::truncdfbf2.
///
//===----------------------------------------------------------------------===//

#define SRC_DOUBLE
#define DST_BFLOAT
#include "fp_trunc.h"

#include "fp_libc_config.h"
#include "shared/builtins/truncdfbf2.h"

extern "C" COMPILER_RT_ABI dst_t __truncdfbf2(src_t a) {
  return __builtin_bit_cast(dst_t, LIBC_NAMESPACE::shared::truncdfbf2(a));
}
