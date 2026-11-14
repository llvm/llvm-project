//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements compiler-rt's __fixunsdfti, truncating double ->
/// __uint128_t conversion (saturating), on top of LLVM-libc's
/// shared::fixunsdfti.
///
//===----------------------------------------------------------------------===//

#include "int_lib.h"

#ifdef CRT_HAS_128BIT
#define DOUBLE_PRECISION
#include "fp_lib.h"

#include "fp_libc_config.h"
#include "shared/builtins/fixunsdfti.h"

extern "C" COMPILER_RT_ABI tu_int __fixunsdfti(fp_t a) {
  return LIBC_NAMESPACE::shared::fixunsdfti(a);
}

#endif // CRT_HAS_128BIT
