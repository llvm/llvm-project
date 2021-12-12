//===-- Implementation of fmaf function -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/math/fmaf.h"
#include "src/__support/common.h"

#include "src/__support/FPUtil/FMA.h"

namespace __llvm_libc {

INLINE_FMA
LLVM_LIBC_FUNCTION(float, fmaf, (float x, float y, float z)) {
  return fputil::fma(x, y, z);
}

} // namespace __llvm_libc
