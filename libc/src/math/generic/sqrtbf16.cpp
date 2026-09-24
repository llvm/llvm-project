//===-- Implementation of sqrtbf16 function -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/math/sqrtbf16.h"
#include "src/__support/math/sqrtbf16.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(bfloat16, sqrtbf16, (bfloat16 x)) {
  return math::sqrtbf16(x);
}

} // namespace LIBC_NAMESPACE_DECL
