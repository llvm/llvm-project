//===-- Implementation of roundbf16 function ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/math/roundbf16.h"
#include "src/__support/math/roundbf16.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(bfloat16, roundbf16, (bfloat16 x)) {
  return math::roundbf16(x);
}

} // namespace LIBC_NAMESPACE_DECL
