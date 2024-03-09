//===-- Implementation header for rintf128 ----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_MATH_RINTF128_H
#define LLVM_LIBC_SRC_MATH_RINTF128_H

#include "src/__support/macros/properties/types.h"

namespace LIBC_NAMESPACE {

float128 rintf128(float128 x);

} // namespace LIBC_NAMESPACE

#endif // LLVM_LIBC_SRC_MATH_RINTF128_H
