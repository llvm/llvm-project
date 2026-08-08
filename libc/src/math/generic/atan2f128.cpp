//===-- Quad-precision atan2 function -------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/math/atan2f128.h"
#include "src/__support/CPP/bit.h"
#include "src/__support/math/atan2f128.h"

#ifndef LIBC_TYPES_HAS_FLOAT128
using float128 = LIBC_NAMESPACE::fputil::Float128;
#endif // LIBC_TYPES_HAS_FLOAT128

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(float128, atan2f128, (float128 y, float128 x)) {
  return cpp::bit_cast<float128>(
      math::atan2f128(Float128(y), Float128(x)).bits);
}

} // namespace LIBC_NAMESPACE_DECL
