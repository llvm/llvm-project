//===-- Implementation of fadd function ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/macros/config.h"

#ifndef LLVM_LIBC_SRC_MATH_FADD_H
#define LLVM_LIBC_SRC_MATH_FADD_H

namespace LIBC_NAMESPACE_DECL {

float fadd(double x, double y);

} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC_MATH_FADD_H
