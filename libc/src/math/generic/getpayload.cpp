//===-- Implementation of getpayload function -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/math/getpayload.h"
#include "src/__support/math/getpayload.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(double, getpayload, (const double *x)) {
  return math::getpayload(x);
}

} // namespace LIBC_NAMESPACE_DECL
