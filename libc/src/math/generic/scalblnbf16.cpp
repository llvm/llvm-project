//===-- Implementation of scalblnbf16 function ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/math/scalblnbf16.h"
#include "src/__support/math/scalblnbf16.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(bfloat16, scalblnbf16, (bfloat16 x, long n)) {
  return math::scalblnbf16(x, n);
}

} // namespace LIBC_NAMESPACE_DECL
