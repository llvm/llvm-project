//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements compiler-rt's __extendhftf2, extend float16 to
/// float128, on top of LLVM-libc's shared::extendhftf2.
///
//===----------------------------------------------------------------------===//

#define QUAD_PRECISION
#include "fp_lib.h"

#include "fp_libc_config.h"
#include "shared/bit.h"
#include "shared/builtins/extendhftf2.h"

#if defined(CRT_HAS_TF_MODE) && defined(COMPILER_RT_HAS_FLOAT16)
#define SRC_HALF
#define DST_QUAD
#include "fp_extend_impl.inc"

extern "C" COMPILER_RT_ABI dst_t __extendhftf2(src_t a) {
  return LIBC_NAMESPACE::shared::extendhftf2(
      LIBC_NAMESPACE::shared::bit_cast<uint16_t>(a));
}
#endif
