//===-- Implementation for tanhbf16(x) function ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/math/tanhbf16.h"
#include "src/__support/common.h"
#include "src/__support/math/tanhbf16.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(bfloat16, tanhbf16, (bfloat16 x)) {
  return math::tanhbf16(x);
}

} // namespace LIBC_NAMESPACE_DECL
