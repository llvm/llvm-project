//===-- Implementation of nextdownl function ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/math/nextdownl.h"
#include "src/__support/math/nextdownl.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(long double, nextdownl, (long double x)) {
  return math::nextdownl(x);
}

} // namespace LIBC_NAMESPACE_DECL
