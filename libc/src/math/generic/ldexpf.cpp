//===-- Implementation of ldexpf function ---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/math/ldexpf.h"
#include "src/__support/math/ldexpf.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(float, ldexpf, (float x, int exp)) {
  return math::ldexpf(x, exp);
}

} // namespace LIBC_NAMESPACE_DECL
