//===-- Single-precision sincos function ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/math/sincosf.h"
#include "src/__support/math/sincosf.h"

namespace LIBC_NAMESPACE_DECL {
LLVM_LIBC_FUNCTION(void, sincosf, (float x, float *sinp, float *cosp)) {
  return math::sincosf(x, sinp, cosp);
}

} // namespace LIBC_NAMESPACE_DECL
