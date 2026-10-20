//===-- Implementation of modf function -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/math/modff.h"
#include "src/__support/math/modff.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(float, modff, (float x, float *iptr)) {
  return math::modff(x, iptr);
}

} // namespace LIBC_NAMESPACE_DECL
