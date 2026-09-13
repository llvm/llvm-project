//===-- Implementation of setpayloadf128 function -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/math/setpayloadf128.h"
#include "src/__support/CPP/bit.h"
#include "src/__support/math/setpayloadf128.h"

namespace LIBC_NAMESPACE_DECL {

using LIBC_NAMESPACE::fputil::Float128;

LLVM_LIBC_FUNCTION(int, setpayloadf128, (float128 * res, float128 pl)) {
  return math::setpayloadf128(cpp::bit_cast<Float128 *>(res),
                              cpp::bit_cast<Float128>(pl));
}

} // namespace LIBC_NAMESPACE_DECL
