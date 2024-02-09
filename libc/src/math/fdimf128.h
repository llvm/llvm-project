//===-- Implementation header for fdimf128 ----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_MATH_FDIMF128_H
#define LLVM_LIBC_SRC_MATH_FDIMF128_H

#include "src/__support/macros/properties/float.h"

namespace LIBC_NAMESPACE {

float128 fdimf128(float128 x, float128 y);

} // namespace LIBC_NAMESPACE

#endif // LLVM_LIBC_SRC_MATH_FDIMF128_H
