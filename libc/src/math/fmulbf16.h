//===-- Implementation header for fmulbf16 ----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_MATH_FMULBF16_H
#define LLVM_LIBC_SRC_MATH_FMULBF16_H

#include "src/__support/FPUtil/bfloat16.h"
#include "src/__support/macros/config.h" // LIBC_NAMESPACE_DECL

namespace LIBC_NAMESPACE_DECL {

bfloat16 fmulbf16(bfloat16 x, bfloat16 y);

} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC_MATH_FMULBF16_H
