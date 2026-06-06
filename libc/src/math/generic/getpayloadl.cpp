//===-- Implementation of getpayloadl function ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/math/getpayloadl.h"
#include "src/__support/math/getpayloadl.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(long double, getpayloadl, (const long double *x)) {
  return math::getpayloadl(x);
}

} // namespace LIBC_NAMESPACE_DECL
