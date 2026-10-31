//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements compiler-rt's __addsf3, single-precision addition, on
/// top of LLVM-libc's shared::addsf3.
///
//===----------------------------------------------------------------------===//

#define SINGLE_PRECISION
#include "fp_lib.h"

#include "fp_libc_config.h"
#include "int_lib.h"
#include "shared/builtins/addsf3.h"

extern "C" COMPILER_RT_ABI fp_t __addsf3(fp_t a, fp_t b) {
  return LIBC_NAMESPACE::shared::addsf3(a, b);
}
