//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements compiler-rt's __fixxfti, truncating long double ->
/// __int128_t conversion (saturating), on top of LLVM-libc's shared::fixxfti.
///
//===----------------------------------------------------------------------===//

#include "int_lib.h"

#include "fp_libc_config.h"
#include "shared/builtins/fixxfti.h"

#if defined(CRT_HAS_128BIT)
extern "C" COMPILER_RT_ABI ti_int __fixxfti(xf_float a) {
  return LIBC_NAMESPACE::shared::fixxfti(a);
}
#endif
