//===-- Single-precision fmodf128 function --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/math/fmodf128.h"
#include "src/__support/math/fmodf128.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(float128, fmodf128, (float128 x, float128 y)) {
  return math::fmodf128(x, y);
}

} // namespace LIBC_NAMESPACE_DECL
