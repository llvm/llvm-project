//===-- Implementation header for frexpf80 ----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_MATH_FREXPF80_H
#define LLVM_LIBC_SRC___SUPPORT_MATH_FREXPF80_H

#include "src/__support/FPUtil/ManipulationFunctions.h"
#include "src/__support/FPUtil/cast.h"
#include "src/__support/FPUtil/float128.h"
#include "src/__support/FPUtil/float80.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {
namespace math {

using LIBC_NAMESPACE::fputil::Float128;
using LIBC_NAMESPACE::fputil::Float80;

// Upcast to Float128 to evaluate frexp cleanly without loss of precision,
// matching the arithmetic operator pattern in FPUtil/float80.h.
LIBC_INLINE constexpr Float80 frexpf80(Float80 x, int *exp) {
  return fputil::cast<Float80>(fputil::frexp(fputil::cast<Float128>(x), *exp));
}

} // namespace math
} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC___SUPPORT_MATH_FREXPF80_H
