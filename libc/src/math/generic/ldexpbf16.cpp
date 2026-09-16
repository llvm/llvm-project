//===-- Implementation of ldexpbf16 function ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/math/ldexpbf16.h"
#include "src/__support/math/ldexpbf16.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(bfloat16, ldexpbf16, (bfloat16 x, int exp)) {
  return math::ldexpbf16(x, exp);
}

} // namespace LIBC_NAMESPACE_DECL
