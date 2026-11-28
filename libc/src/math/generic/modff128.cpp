//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Implementation of the modff128 function.
///
//===----------------------------------------------------------------------===//

#include "src/math/modff128.h"
#include "src/__support/CPP/bit.h"
#include "src/__support/math/modff128.h"

namespace LIBC_NAMESPACE_DECL {

using LIBC_NAMESPACE::fputil::Float128;

LLVM_LIBC_FUNCTION(float128, modff128, (float128 x, float128 *iptr)) {
  Float128 iptr_val{};
  float128 result = cpp::bit_cast<float128>(
      math::modff128(cpp::bit_cast<Float128>(x), &iptr_val));
  *iptr = cpp::bit_cast<float128>(iptr_val);
  return result;
}

} // namespace LIBC_NAMESPACE_DECL
