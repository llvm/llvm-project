//===-- Implementation of ceilf128 function -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/math/ceilf128.h"
#include "src/__support/math/ceilf128.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(float128, ceilf128, (float128 x)) {
  return math::ceilf128(x);
}

} // namespace LIBC_NAMESPACE_DECL
