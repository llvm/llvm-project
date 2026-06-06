//===-- Implementation of ldexpf16 function -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/math/ldexpf16.h"

#include "src/__support/math/ldexpf16.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(float16, ldexpf16, (float16 x, int exp)) {
  return math::ldexpf16(x, exp);
}

} // namespace LIBC_NAMESPACE_DECL
