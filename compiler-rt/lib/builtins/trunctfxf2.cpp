//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements compiler-rt's __trunctfxf2, truncate float128 to long
/// double, on top of LLVM-libc's shared::trunctfxf2.
///
//===----------------------------------------------------------------------===//

#define QUAD_PRECISION
#include "fp_lib.h"

#include "fp_libc_config.h"
#include "shared/builtins/trunctfxf2.h"

#if defined(CRT_HAS_TF_MODE) && __LDBL_MANT_DIG__ == 64 && defined(__x86_64__)
extern "C" COMPILER_RT_ABI xf_float __trunctfxf2(tf_float a) {
  return LIBC_NAMESPACE::shared::trunctfxf2(a);
}
#endif
