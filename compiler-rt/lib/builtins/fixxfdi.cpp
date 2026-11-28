//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements compiler-rt's __fixxfdi, truncating long double ->
/// int64_t conversion (saturating), on top of LLVM-libc's shared::fixxfdi.
///
//===----------------------------------------------------------------------===//

#include "int_lib.h"

#include "fp_libc_config.h"
#include "shared/builtins/fixxfdi.h"

#if !_ARCH_PPC
extern "C" COMPILER_RT_ABI di_int __fixxfdi(xf_float a) {
  return LIBC_NAMESPACE::shared::fixxfdi(a);
}
#endif
