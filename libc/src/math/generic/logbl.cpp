//===-- Implementation of logbl function ---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/math/logbl.h"
#include "src/__support/math/logbl.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(long double, logbl, (long double x)) {
  return math::logbl(x);
}

} // namespace LIBC_NAMESPACE_DECL
