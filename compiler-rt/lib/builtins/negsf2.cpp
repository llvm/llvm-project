//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
/// This file implements compiler-rt's __negsf2, single-precision negation,
/// on top of LLVM-libc's shared::negsf2.
///
//===----------------------------------------------------------------------===//

#define SINGLE_PRECISION
#include "fp_lib.h"

#include "fp_libc_config.h"
#include "int_lib.h"
#include "shared/builtins/negsf2.h"

extern "C" COMPILER_RT_ABI fp_t __negsf2(fp_t a) {
  return LIBC_NAMESPACE::shared::negsf2(a);
}
