//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements compiler-rt's __trunctfdf2, truncate float128 to
/// double, on top of LLVM-libc's shared::trunctfdf2.
///
//===----------------------------------------------------------------------===//

#define QUAD_PRECISION
#include "fp_lib.h"

#include "fp_libc_config.h"
#include "shared/builtins/trunctfdf2.h"

#ifdef CRT_HAS_TF_MODE
extern "C" COMPILER_RT_ABI double __trunctfdf2(tf_float a) {
  return LIBC_NAMESPACE::shared::trunctfdf2(a);
}
#endif
