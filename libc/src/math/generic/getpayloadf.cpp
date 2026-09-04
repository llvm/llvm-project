//===-- Implementation of getpayloadf function ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/math/getpayloadf.h"
#include "src/__support/math/getpayloadf.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(float, getpayloadf, (const float *x)) {
  return math::getpayloadf(x);
}

} // namespace LIBC_NAMESPACE_DECL
