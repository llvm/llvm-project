//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements compiler-rt's __extendsfdf2, extend float to double, on
/// top of LLVM-libc's shared::extendsfdf2.
///
//===----------------------------------------------------------------------===//

#include "shared/builtins/extendsfdf2.h"
#include "fp_libc_config.h"
#include "int_lib.h"

extern "C" COMPILER_RT_ABI double __extendsfdf2(float a) {
  return LIBC_NAMESPACE::shared::extendsfdf2(a);
}
