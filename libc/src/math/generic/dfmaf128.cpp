//===-- Implementation of dfmaf128 function -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/math/dfmaf128.h"
#include "src/__support/math/dfmaf128.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(double, dfmaf128, (float128 x, float128 y, float128 z)) {
  return math::dfmaf128(x, y, z);
}

} // namespace LIBC_NAMESPACE_DECL
