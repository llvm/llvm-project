//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements compiler-rt's __floatdixf, int64_t -> long double
/// conversion (round to nearest), on top of LLVM-libc's shared::floatdixf.
///
//===----------------------------------------------------------------------===//

#include "int_lib.h"

#include "fp_libc_config.h"
#include "shared/builtins/floatdixf.h"

#if !_ARCH_PPC
extern "C" COMPILER_RT_ABI xf_float __floatdixf(di_int a) {
  return LIBC_NAMESPACE::shared::floatdixf(a);
}
#endif
