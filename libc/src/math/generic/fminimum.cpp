//===-- Implementation of fminimum function--------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/math/fminimum.h"
#include "src/__support/math/fminimum.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(double, fminimum, (double x, double y)) {
  return math::fminimum(x, y);
}

} // namespace LIBC_NAMESPACE_DECL
