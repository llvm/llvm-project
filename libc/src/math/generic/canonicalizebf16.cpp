//===-- Implementation of canonicalizebf16 function -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/math/canonicalizebf16.h"
#include "src/__support/math/canonicalizebf16.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(int, canonicalizebf16, (bfloat16 * cx, const bfloat16 *x)) {
  return math::canonicalizebf16(cx, x);
}

} // namespace LIBC_NAMESPACE_DECL
