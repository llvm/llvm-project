//===-- Implementation of fdimf function ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/math/fdimf.h"
#include "src/__support/math/fdimf.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(float, fdimf, (float x, float y)) {
  LIBC_FENV_ACCESS_ON
  return math::fdimf(x, y);
}

} // namespace LIBC_NAMESPACE_DECL
