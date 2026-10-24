//===-- Implementation header for fabsbf16 ----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_MATH_FABSBF16_H
#define LLVM_LIBC_SRC_MATH_FABSBF16_H

#include "src/__support/macros/config.h"
#include "src/__support/macros/properties/types.h"

namespace LIBC_NAMESPACE_DECL {

#if defined(LIBC_TYPES_HAS_BUILTIN_BFLOAT16)
__bf16 fabsbf16(__bf16 x);
#else
bfloat16 fabsbf16(bfloat16 x);
#endif

} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC_MATH_FABSBF16_H
