//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Implementation of the inverse hyperbolic sine function for bfloat16. The
/// input is converted to float, asinhf is evaluated, and the result is rounded
/// once to bfloat16.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_MATH_ASINHBF16_H
#define LLVM_LIBC_SRC___SUPPORT_MATH_ASINHBF16_H

#include "src/__support/FPUtil/bfloat16.h"
#include "src/__support/FPUtil/cast.h"
#include "src/__support/macros/config.h"
#include "src/__support/math/asinhf.h"

namespace LIBC_NAMESPACE_DECL {
namespace math {

LIBC_INLINE bfloat16 asinhbf16(bfloat16 x) {
  // Evaluate in float to use the established asinh implementation, then round
  // once to the narrower bfloat16 result type.
  return fputil::cast<bfloat16>(asinhf(static_cast<float>(x)));
}

} // namespace math
} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC___SUPPORT_MATH_ASINHBF16_H
