//===-- Implementation of fmaximum function--------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/math/fmaximum.h"
#include "src/__support/math/fmaximum.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(double, fmaximum, (double x, double y)) {
  return math::fmaximum(x, y);
}

} // namespace LIBC_NAMESPACE_DECL
