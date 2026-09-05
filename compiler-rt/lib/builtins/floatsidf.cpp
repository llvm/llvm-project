//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements compiler-rt's __floatsidf, int32_t -> double conversion
/// (round to nearest), on top of LLVM-libc's shared::floatsidf.
///
//===----------------------------------------------------------------------===//

#define DOUBLE_PRECISION
#include "fp_lib.h"

#include "fp_libc_config.h"
#include "int_lib.h"
#include "shared/builtins/floatsidf.h"

extern "C" COMPILER_RT_ABI fp_t __floatsidf(si_int a) {
  return LIBC_NAMESPACE::shared::floatsidf(a);
}
