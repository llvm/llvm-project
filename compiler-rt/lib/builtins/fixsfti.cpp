//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements compiler-rt's __fixsfti, truncating float -> __int128_t
/// conversion (saturating), on top of LLVM-libc's shared::fixsfti.
///
//===----------------------------------------------------------------------===//

#include "int_lib.h"

#ifdef CRT_HAS_128BIT
#define SINGLE_PRECISION
#include "fp_lib.h"

#include "fp_libc_config.h"
#include "shared/builtins/fixsfti.h"

extern "C" COMPILER_RT_ABI ti_int __fixsfti(fp_t a) {
  return LIBC_NAMESPACE::shared::fixsfti(a);
}

#endif // CRT_HAS_128BIT
