//===-- Implementation of the fmax function for GPU -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/math/fmax.h"

#include "src/__support/CPP/bit.h"
#include "src/__support/common.h"
#include "src/__support/macros/optimization.h"

namespace LIBC_NAMESPACE {

LLVM_LIBC_FUNCTION(double, fmax, (double x, double y)) {
  // FIXME: The builtin function does not correctly handle the +/-0.0 case.
  if (LIBC_UNLIKELY(x == y))
    return cpp::bit_cast<double>(cpp::bit_cast<uint64_t>(x) &
                                 cpp::bit_cast<uint64_t>(y));
  return __builtin_fmax(x, y);
}

} // namespace LIBC_NAMESPACE
