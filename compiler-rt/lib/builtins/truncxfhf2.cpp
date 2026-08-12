//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements compiler-rt's __truncxfhf2, truncate long double to
/// float16, on top of LLVM-libc's shared::truncxfhf2.
///
//===----------------------------------------------------------------------===//

#include "int_lib.h"
#define SRC_SINGLE
#define DST_HALF
#include "fp_trunc.h"

#include "fp_libc_config.h"
#include "shared/builtins/truncxfhf2.h"

extern "C" COMPILER_RT_ABI dst_t __truncxfhf2(xf_float a) {
  return __builtin_bit_cast(dst_t, LIBC_NAMESPACE::shared::truncxfhf2(a));
}
