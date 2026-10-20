//===-- Implementation of scalblnf128 function ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/math/scalblnf128.h"
#include "src/__support/math/scalblnf128.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(float128, scalblnf128, (float128 x, long n)) {
  return math::scalblnf128(x, n);
}

} // namespace LIBC_NAMESPACE_DECL
