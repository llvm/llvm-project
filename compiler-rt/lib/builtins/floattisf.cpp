//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements compiler-rt's __floattisf, __int128_t -> float
/// conversion (round to nearest), on top of LLVM-libc's shared::floattisf.
///
//===----------------------------------------------------------------------===//

#include "int_lib.h"

#ifdef CRT_HAS_128BIT
#define SINGLE_PRECISION
#include "fp_lib.h"

#include "fp_libc_config.h"
#include "shared/builtins/floattisf.h"

extern "C" COMPILER_RT_ABI fp_t __floattisf(ti_int a) {
  return LIBC_NAMESPACE::shared::floattisf(a);
}

#endif // CRT_HAS_128BIT
