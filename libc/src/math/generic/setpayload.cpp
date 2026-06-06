//===-- Implementation of setpayload function -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/math/setpayload.h"
#include "src/__support/math/setpayload.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(int, setpayload, (double *res, double pl)) {
  return math::setpayload(res, pl);
}

} // namespace LIBC_NAMESPACE_DECL
