//===-- Quad-precision atan2 function -------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/math/atan2f128.h"
#include "src/__support/math/atan2f128.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(float128, atan2f128, (float128 y, float128 x)) {
  return math::atan2f128(y, x);
}

} // namespace LIBC_NAMESPACE_DECL
