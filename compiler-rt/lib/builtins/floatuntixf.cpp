//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements compiler-rt's __floatuntixf, __uint128_t -> long double
/// conversion (round to nearest), on top of LLVM-libc's shared::floatuntixf.
///
//===----------------------------------------------------------------------===//

#include "int_lib.h"

#include "fp_libc_config.h"
#include "shared/builtins/floatuntixf.h"

#if defined(CRT_HAS_128BIT)
extern "C" COMPILER_RT_ABI xf_float __floatuntixf(tu_int a) {
  return LIBC_NAMESPACE::shared::floatuntixf(a);
}
#endif
