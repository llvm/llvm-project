//===-- Implementation of f16subl function --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/math/f16subl.h"
#include "src/__support/math/f16subl.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(float16, f16subl, (long double x, long double y)) {
  return math::f16subl(x, y);
}

} // namespace LIBC_NAMESPACE_DECL
