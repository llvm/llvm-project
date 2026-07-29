//===-- Implementation of remquol function --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/math/remquol.h"
#include "src/__support/math/remquol.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(long double, remquol,
                   (long double x, long double y, int *exp)) {
  return math::remquol(x, y, exp);
}

} // namespace LIBC_NAMESPACE_DECL
