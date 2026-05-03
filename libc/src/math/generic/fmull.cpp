//===-- Implementation of fmull function ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/math/fmull.h"
#include "src/__support/math/fmull.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(float, fmull, (long double x, long double y)) {
  return math::fmull(x, y);
}

} // namespace LIBC_NAMESPACE_DECL
