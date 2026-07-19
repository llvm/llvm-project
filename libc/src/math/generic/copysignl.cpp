//===-- Implementation of copysignl function ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/math/copysignl.h"
#include "src/__support/math/copysignl.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(long double, copysignl, (long double x, long double y)) {
  return math::copysignl(x, y);
}

} // namespace LIBC_NAMESPACE_DECL
