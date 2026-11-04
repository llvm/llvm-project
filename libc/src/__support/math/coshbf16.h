//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Implementation of the coshbf16 function, which computes the hyperbolic
/// cosine of a bfloat16 value.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_MATH_COSHBF16_H
#define LLVM_LIBC_SRC___SUPPORT_MATH_COSHBF16_H

#include "src/__support/FPUtil/bfloat16.h"
#include "src/__support/macros/config.h"
#include "src/__support/macros/properties/types.h"

namespace LIBC_NAMESPACE_DECL {
namespace math {

LIBC_INLINE bfloat16 coshbf16(bfloat16 x) {
  // TODO: Implement coshbf16.
  return x;
}

} // namespace math
} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC___SUPPORT_MATH_COSHBF16_H
