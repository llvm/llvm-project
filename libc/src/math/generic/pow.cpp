//===-- Implementation of pow function ------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/math/pow.h"
#include "src/math/expf.h"
#include "src/math/logf.h"

#include "src/__support/common.h"

namespace __llvm_libc {

LLVM_LIBC_FUNCTION(double, pow, (double x, double y)) {
  // Place-holder implementation for double precision pow function.
  // TODO: Implement correctly rounded pow function for double precision.
  return static_cast<double>(
      expf(static_cast<float>(y) * logf(static_cast<float>(x))));
}

} // namespace __llvm_libc
