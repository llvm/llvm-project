//===-- Implementation header for llogbl ------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_MATH_LLOGBL_H
#define LLVM_LIBC_SRC_MATH_LLOGBL_H

#include "src/__support/FPUtil/ManipulationFunctions.h"
#include "src/__support/macros/config.h"
#include "src/__support/macros/properties/types.h"

namespace LIBC_NAMESPACE_DECL {

// Inline constexpr implementation: extract the unbiased exponent of a long double
// by delegating to the existing constexpr template fputil::intlogb<long>.
LIBC_INLINE constexpr long llogbl(long double x) {
  return fputil::intlogb<long>(x);
}

} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC_MATH_LLOGBL_H
