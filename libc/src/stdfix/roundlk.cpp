//===-- Implementation of roundlk function --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "roundlk.h"
#include "src/__support/common.h"
#include "src/__support/fixed_point/fx_bits.h"

namespace LIBC_NAMESPACE {

LLVM_LIBC_FUNCTION(long accum, roundlk, (long accum x, int n)) {
  return fixed_point::round(x, n);
}

} // namespace LIBC_NAMESPACE
