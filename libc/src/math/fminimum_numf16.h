//===-- Implementation header for fminimum_numf16 ---------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_MATH_FMINIMUM_NUMF16_H
#define LLVM_LIBC_SRC_MATH_FMINIMUM_NUMF16_H

#include "src/__support/macros/properties/types.h"

namespace LIBC_NAMESPACE {

float16 fminimum_numf16(float16 x, float16 y);

} // namespace LIBC_NAMESPACE

#endif // LLVM_LIBC_SRC_MATH_FMINIMUM_NUMF16_H
