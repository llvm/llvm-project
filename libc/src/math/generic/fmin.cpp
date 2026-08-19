//===-- Implementation of fmin function -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/math/fmin.h"
#include "src/__support/math/fmin.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(double, fmin, (double x, double y)) {
  return math::fmin(x, y);
}

} // namespace LIBC_NAMESPACE_DECL
