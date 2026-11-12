//===-- Implementation of setpayloadf function ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/math/setpayloadf.h"
#include "src/__support/math/setpayloadf.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(int, setpayloadf, (float *res, float pl)) {
  return math::setpayloadf(res, pl);
}

} // namespace LIBC_NAMESPACE_DECL
