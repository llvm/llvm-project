//===-- Double-precision asin function ------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/math/asin.h"
#include "src/math/asinf.h"

#include "src/__support/common.h"

namespace __llvm_libc {

LLVM_LIBC_FUNCTION(double, asin, (double x)) {
  // Place-holder implementation for double precision asin function.
  // TODO: Implement correctly rounded asin function for double precision.
  return static_cast<double>(asinf(static_cast<float>(x)));
}

} // namespace __llvm_libc
