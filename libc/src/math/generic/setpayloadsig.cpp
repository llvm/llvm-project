//===-- Implementation of setpayloadsig function --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/math/setpayloadsig.h"
#include "src/__support/math/setpayloadsig.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(int, setpayloadsig, (double *res, double pl)) {
  return math::setpayloadsig(res, pl);
}

} // namespace LIBC_NAMESPACE_DECL
