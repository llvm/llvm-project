//===-- Implementation for atanbf16(x) function ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception.
//
//===----------------------------------------------------------------------===//

#include "src/math/atanbf16.h"
#include "src/__support/math/atanbf16.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(bfloat16, atanbf16, (bfloat16 x)) {
  return math::atanbf16(x);
}

} // namespace LIBC_NAMESPACE_DECL
