//===-- Implementation of modff16 function --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/math/modff16.h"
#include "src/__support/math/modff16.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(float16, modff16, (float16 x, float16 *iptr)) {
  return math::modff16(x, iptr);
}

} // namespace LIBC_NAMESPACE_DECL
