//===-- Implementation header for ceilf128 ----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_MATH_CEILF128_H
#define LLVM_LIBC_SRC___SUPPORT_MATH_CEILF128_H

#include "src/__support/FPUtil/NearestIntegerOperations.h"
#include "src/__support/FPUtil/float128.h"
#include "src/__support/macros/config.h"
#include "src/__support/macros/properties/types.h"

using LIBC_NAMESPACE::fputil::Float128;

namespace LIBC_NAMESPACE_DECL {
namespace math {

#ifdef LIBC_TYPES_HAS_FLOAT128
LIBC_INLINE constexpr float128 ceilf128(float128 x) {
  return static_cast<float128>(fputil::ceil(fputil::Float128(x)));
}
#else
LIBC_INLINE constexpr fputil::Float128 ceilf128(fputil::Float128 x) {
  return fputil::ceil(x);
}
#endif // LIBC_TYPES_HAS_FLOAT128

} // namespace math
} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC___SUPPORT_MATH_CEILF128_H
