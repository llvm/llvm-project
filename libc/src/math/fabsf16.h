//===-- Implementation header for fabsf16 -----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_MATH_FABSF16_H
#define LLVM_LIBC_SRC_MATH_FABSF16_H

#include "src/__support/FPUtil/float16.h"
#include "src/__support/macros/config.h"
#include "src/__support/macros/properties/types.h"

namespace LIBC_NAMESPACE_DECL {

#ifndef LIBC_TYPES_HAS_FLOAT16
using float16 = LIBC_NAMESPACE::fputil::Float16;
#endif // LIBC_TYPES_HAS_FLOAT16

float16 fabsf16(float16 x);

} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC_MATH_FABSF16_H
