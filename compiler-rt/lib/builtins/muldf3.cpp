//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements compiler-rt's __muldf3, double-precision
/// multiplication, on top of LLVM-libc's shared::muldf3.
///
//===----------------------------------------------------------------------===//

#define DOUBLE_PRECISION
#include "fp_lib.h"

#include "fp_libc_config.h"
#include "int_lib.h"
#include "shared/builtins/muldf3.h"

extern "C" COMPILER_RT_ABI fp_t __muldf3(fp_t a, fp_t b) {
  return LIBC_NAMESPACE::shared::muldf3(a, b);
}
