//===-- Implementation header for bf16addf ----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_MATH_BF16ADDF_H
#define LLVM_LIBC_SRC_MATH_BF16ADDF_H

#include "src/__support/macros/config.h"
#include "src/__support/macros/properties/types.h"

namespace LIBC_NAMESPACE_DECL {

bfloat16 bf16addf(float x, float y);

} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC_MATH_BF16ADDF_H
