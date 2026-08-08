//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements compiler-rt's __fixtfti, truncating float128 ->
/// __int128_t conversion (saturating), on top of LLVM-libc's shared::fixtfti.
///
//===----------------------------------------------------------------------===//

#define QUAD_PRECISION
#include "fp_lib.h"

#include "fp_libc_config.h"
#include "int_lib.h"
#include "shared/builtins/fixtfti.h"

#if defined(CRT_HAS_TF_MODE)
extern "C" COMPILER_RT_ABI ti_int __fixtfti(fp_t a) {
  return LIBC_NAMESPACE::shared::fixtfti(a);
}
#endif
