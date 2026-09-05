//===-- Implementation of totalordermagf128 function ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/math/totalordermagf128.h"
#include "src/__support/CPP/bit.h"
#include "src/__support/math/totalordermagf128.h"

namespace LIBC_NAMESPACE_DECL {

using LIBC_NAMESPACE::fputil::Float128;

LLVM_LIBC_FUNCTION(int, totalordermagf128,
                   (const float128 *x, const float128 *y)) {
#ifdef LIBC_TYPES_HAS_NATIVE_FLOAT128
  Float128 x_f128 = cpp::bit_cast<Float128>(*x);
  Float128 y_f128 = cpp::bit_cast<Float128>(*y);
  return math::totalordermagf128(&x_f128, &y_f128);
#else
  return math::totalordermagf128(x, y);
#endif
}

} // namespace LIBC_NAMESPACE_DECL
