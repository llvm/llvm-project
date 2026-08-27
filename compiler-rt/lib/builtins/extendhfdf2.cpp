//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements compiler-rt's __extendhfdf2, extend float16 to double,
/// on top of LLVM-libc's shared::extendhfdf2.
///
//===----------------------------------------------------------------------===//

#define SRC_HALF
#define DST_DOUBLE
#include "fp_extend_impl.inc"

#include "fp_libc_config.h"
#include "shared/bit.h"
#include "shared/builtins/extendhfdf2.h"

extern "C" COMPILER_RT_ABI NOINLINE dst_t __extendhfdf2(src_t a) {
  return LIBC_NAMESPACE::shared::extendhfdf2(
      LIBC_NAMESPACE::shared::bit_cast<uint16_t>(a));
}
