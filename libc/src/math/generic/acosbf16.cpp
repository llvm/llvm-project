//===-- Implementation of acosbf16 function -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception.
//
//
//===----------------------------------------------------------------------===//

#include "src/math/acosbf16.h"
#include "src/__support/math/acosbf16.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(bfloat16, acosbf16, (bfloat16 x)) {
  return math::acosbf16(x);
}

} // namespace LIBC_NAMESPACE_DECL
