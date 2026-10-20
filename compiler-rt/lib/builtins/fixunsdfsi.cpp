//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements compiler-rt's __fixunsdfsi, truncating double ->
/// uint32_t conversion (saturating), on top of LLVM-libc's shared::fixunsdfsi.
///
//===----------------------------------------------------------------------===//

#define DOUBLE_PRECISION
#include "fp_lib.h"

#include "fp_libc_config.h"
#include "int_lib.h"
#include "shared/builtins/fixunsdfsi.h"

extern "C" COMPILER_RT_ABI su_int __fixunsdfsi(fp_t a) {
  return LIBC_NAMESPACE::shared::fixunsdfsi(a);
}
