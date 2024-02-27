//===-- Implementation of the fminf function for GPU ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/math/fminf.h"

#include "src/__support/CPP/bit.h"
#include "src/__support/common.h"
#include "src/__support/macros/optimization.h"

namespace LIBC_NAMESPACE {

LLVM_LIBC_FUNCTION(float, fminf, (float x, float y)) {
  // FIXME: The builtin function does not correctly handle the +/-0.0 case.
  if (LIBC_UNLIKELY(x == y))
    return cpp::bit_cast<float>(cpp::bit_cast<uint32_t>(x) |
                                cpp::bit_cast<uint32_t>(y));
  return __builtin_fminf(x, y);
}

} // namespace LIBC_NAMESPACE
