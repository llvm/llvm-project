//===-- Implementation of fminimumf16 function ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/math/fminimumf16.h"
#include "src/__support/math/fminimumf16.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(float16, fminimumf16, (float16 x, float16 y)) {
  return math::fminimumf16(x, y);
}

} // namespace LIBC_NAMESPACE_DECL
