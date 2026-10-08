//===-- Implementation of scalblnl function -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/math/scalblnl.h"
#include "src/__support/math/scalblnl.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(long double, scalblnl, (long double x, long n)) {
  return math::scalblnl(x, n);
}

} // namespace LIBC_NAMESPACE_DECL
