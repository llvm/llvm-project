//===-- Implementation of totalorder function -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/math/totalorder.h"
#include "src/__support/math/totalorder.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(int, totalorder, (const double *x, const double *y)) {
  return math::totalorder(x, y);
}

} // namespace LIBC_NAMESPACE_DECL
