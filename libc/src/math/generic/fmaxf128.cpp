//===-- Implementation of fmaxf128 function -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/math/fmaxf128.h"
#include "src/__support/CPP/bit.h"
#include "src/__support/math/fmaxf128.h"

namespace LIBC_NAMESPACE_DECL {

using LIBC_NAMESPACE::fputil::Float128;

LLVM_LIBC_FUNCTION(float128, fmaxf128, (float128 x, float128 y)) {
  return cpp::bit_cast<float128>(
      math::fmaxf128(cpp::bit_cast<Float128>(x), cpp::bit_cast<Float128>(y)));
}

} // namespace LIBC_NAMESPACE_DECL
