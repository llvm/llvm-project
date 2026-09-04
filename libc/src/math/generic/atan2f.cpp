//===-- Single-precision atan2f function ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/math/atan2f.h"
#include "src/__support/math/atan2f.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(float, atan2f, (float y, float x)) {
  return math::atan2f(y, x);
}

} // namespace LIBC_NAMESPACE_DECL
