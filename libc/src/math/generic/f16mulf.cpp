//===-- Implementation of f16mulf function --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/math/f16mulf.h"
#include "src/__support/math/f16mulf.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(float16, f16mulf, (float x, float y)) {
  return math::f16mulf(x, y);
}

} // namespace LIBC_NAMESPACE_DECL
