//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements compiler-rt's __fixdfti, truncating double ->
/// __int128_t conversion (saturating), on top of LLVM-libc's shared::fixdfti.
///
//===----------------------------------------------------------------------===//

#include "int_lib.h"

#ifdef CRT_HAS_128BIT
#define DOUBLE_PRECISION
#include "fp_lib.h"

#include "fp_libc_config.h"
#include "shared/builtins/fixdfti.h"

extern "C" COMPILER_RT_ABI ti_int __fixdfti(fp_t a) {
  return LIBC_NAMESPACE::shared::fixdfti(a);
}

#endif // CRT_HAS_128BIT
