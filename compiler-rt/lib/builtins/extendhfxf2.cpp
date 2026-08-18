//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements compiler-rt's __extendhfxf2, extend float16 to long
/// double, on top of LLVM-libc's shared::extendhfxf2.
///
//===----------------------------------------------------------------------===//

#include "int_lib.h"
#define SRC_HALF
#define DST_DOUBLE
#include "fp_extend.h"

#include "fp_libc_config.h"
#include "shared/builtins/extendhfxf2.h"

extern "C" COMPILER_RT_ABI xf_float __extendhfxf2(src_t a) {
  return LIBC_NAMESPACE::shared::extendhfxf2(__builtin_bit_cast(uint16_t, a));
}
