//===-- Implementation header for f16sqrtf ----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_MATH_F16SQRTF_H
#define LLVM_LIBC_SRC_MATH_F16SQRTF_H

#include "src/__support/macros/properties/types.h"

namespace LIBC_NAMESPACE {

float16 f16sqrtf(float x);

} // namespace LIBC_NAMESPACE

#endif // LLVM_LIBC_SRC_MATH_F16SQRTF_H
