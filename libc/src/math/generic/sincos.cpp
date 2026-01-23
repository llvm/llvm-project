//===-- Double-precision sincos function ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/math/sincos.h"
#include "src/__support/math/sincos.h"

namespace LIBC_NAMESPACE_DECL {
LLVM_LIBC_FUNCTION(void, sincos, (double x, double *sin_x, double *cos_x)) {
  return math::sincos(x, sin_x, cos_x);
}

} // namespace LIBC_NAMESPACE_DECL
