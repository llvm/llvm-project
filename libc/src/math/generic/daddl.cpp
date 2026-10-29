//===-- Implementation of daddl function ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/math/daddl.h"
#include "src/__support/math/daddl.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(double, daddl, (long double x, long double y)) {
  return math::daddl(x, y);
}

} // namespace LIBC_NAMESPACE_DECL
