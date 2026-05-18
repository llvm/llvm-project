//===-- Implementation of dsubl function ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/math/dsubl.h"
#include "src/__support/math/dsubl.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(double, dsubl, (long double x, long double y)) {
  return math::dsubl(x, y);
}

} // namespace LIBC_NAMESPACE_DECL
