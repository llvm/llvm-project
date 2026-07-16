//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements compiler-rt's __trunctfsf2, truncate float128 to float,
/// on top of LLVM-libc's shared::trunctfsf2.
///
//===----------------------------------------------------------------------===//

#define QUAD_PRECISION
#include "fp_lib.h"

#include "fp_libc_config.h"
#include "shared/builtins/trunctfsf2.h"

#if defined(CRT_HAS_TF_MODE)
extern "C" COMPILER_RT_ABI float __trunctfsf2(tf_float a) {
  return LIBC_NAMESPACE::shared::trunctfsf2(a);
}
#endif
