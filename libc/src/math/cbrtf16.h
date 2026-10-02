//===-- Implementation header for cbrtf16 -----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_MATH_CBRTF16_H
#define LLVM_LIBC_SRC_MATH_CBRTF16_H

#include "src/__support/macros/config.h"           // LIBC_NAMESPACE_DECL
#include "src/__support/macros/properties/types.h" // float16

namespace LIBC_NAMESPACE_DECL {

float16 cbrtf16(float16 x);

} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC_MATH_CBRTF16_H
