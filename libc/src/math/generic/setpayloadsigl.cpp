//===-- Implementation of setpayloadsigl function -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/math/setpayloadsigl.h"
#include "src/__support/math/setpayloadsigl.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(int, setpayloadsigl, (long double *res, long double pl)) {
  return math::setpayloadsigl(res, pl);
}

} // namespace LIBC_NAMESPACE_DECL
