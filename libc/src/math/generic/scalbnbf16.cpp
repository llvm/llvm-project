//===-- Implementation of scalbnbf16 function -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/math/scalbnbf16.h"
#include "src/__support/math/scalbnbf16.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(bfloat16, scalbnbf16, (bfloat16 x, int n)) {
  return math::scalbnbf16(x, n);
}

} // namespace LIBC_NAMESPACE_DECL
