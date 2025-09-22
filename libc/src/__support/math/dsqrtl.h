//===-- Implementation header for dsqrtl ------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_MATH_DSQRTL_H
#define LLVM_LIBC_SRC___SUPPORT_MATH_DSQRTL_H

#include "src/__support/FPUtil/generic/sqrt.h"

namespace LIBC_NAMESPACE_DECL {

namespace math {

LIBC_INLINE static constexpr double dsqrtl(long double x) {
  return fputil::sqrt<double>(x);
}

} // namespace math

} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC___SUPPORT_MATH_DSQRTL_H
