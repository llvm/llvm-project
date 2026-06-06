//===-- Double-precision x^y function -------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/math/pow.h"
#include "src/__support/math/pow.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(double, pow, (double x, double y)) {
  return math::pow(x, y);
}

} // namespace LIBC_NAMESPACE_DECL
