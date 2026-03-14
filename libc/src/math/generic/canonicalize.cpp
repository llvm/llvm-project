//===-- Implementation of canonicalize function----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/math/canonicalize.h"
#include "src/__support/math/canonicalize.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(int, canonicalize, (double *cx, const double *x)) {
  return math::canonicalize(cx, x);
}

} // namespace LIBC_NAMESPACE_DECL
