//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements compiler-rt's __extenddftf2, extend double to float128,
/// on top of LLVM-libc's shared::extenddftf2.
///
//===----------------------------------------------------------------------===//

#define QUAD_PRECISION
#include "fp_lib.h"

#include "fp_libc_config.h"
#include "shared/builtins/extenddftf2.h"

#if defined(CRT_HAS_TF_MODE)
extern "C" COMPILER_RT_ABI tf_float __extenddftf2(double a) {
  return LIBC_NAMESPACE::shared::extenddftf2(a);
}
#endif
