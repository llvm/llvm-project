//===-- Implementation of ldexpf128 function ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/math/ldexpf128.h"

#include "src/__support/math/ldexpf128.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(float128, ldexpf128, (float128 x, int exp)) {
  return math::ldexpf128(x, exp);
}

} // namespace LIBC_NAMESPACE_DECL
