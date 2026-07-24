//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements compiler-rt's __extendbfsf2, extend bfloat16 to float,
/// on top of LLVM-libc's shared::extendbfsf2.
///
//===----------------------------------------------------------------------===//

#define SRC_BFLOAT16
#define DST_SINGLE
#include "fp_extend.h"

#include "fp_libc_config.h"
#include "shared/builtins/extendbfsf2.h"

extern "C" COMPILER_RT_ABI dst_t __extendbfsf2(src_t a) {
  return LIBC_NAMESPACE::shared::extendbfsf2(__builtin_bit_cast(uint16_t, a));
}
