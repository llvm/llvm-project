//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
/// This file implements compiler-rt's __negdf2, double-precision negation,
/// on top of LLVM-libc's shared::negdf2.
///
//===----------------------------------------------------------------------===//

#define DOUBLE_PRECISION
#include "fp_lib.h"

#include "fp_libc_config.h"
#include "int_lib.h"
#include "shared/builtins/negdf2.h"

extern "C" COMPILER_RT_ABI fp_t __negdf2(fp_t a) {
  return LIBC_NAMESPACE::shared::negdf2(a);
}
