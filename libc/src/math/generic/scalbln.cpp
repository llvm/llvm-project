//===-- Implementation of scalbln function --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/math/scalbln.h"
#include "src/__support/math/scalbln.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(double, scalbln, (double x, long n)) {
  return math::scalbln(x, n);
}

} // namespace LIBC_NAMESPACE_DECL
