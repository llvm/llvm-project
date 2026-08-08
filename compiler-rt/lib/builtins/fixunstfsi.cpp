//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements compiler-rt's __fixunstfsi, truncating float128 ->
/// uint32_t conversion (saturating), on top of LLVM-libc's shared::fixunstfsi.
///
//===----------------------------------------------------------------------===//

#define QUAD_PRECISION
#include "fp_lib.h"

#include "fp_libc_config.h"
#include "int_lib.h"
#include "shared/builtins/fixunstfsi.h"

#if defined(CRT_HAS_TF_MODE)
extern "C" COMPILER_RT_ABI su_int __fixunstfsi(fp_t a) {
  return LIBC_NAMESPACE::shared::fixunstfsi(a);
}
#endif
