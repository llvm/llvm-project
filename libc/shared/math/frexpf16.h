//===-- Shared frexpf16 function --------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SHARED_MATH_FREXPF16_H
#define LLVM_LIBC_SHARED_MATH_FREXPF16_H

#include "include/llvm-libc-macros/float16-macros.h"
#include "shared/libc_common.h"

#ifdef LIBC_TYPES_HAS_FLOAT16

#include "src/__support/math/frexpf16.h"

namespace LIBC_NAMESPACE_DECL {
namespace shared {

using math::frexpf16;

} // namespace shared
} // namespace LIBC_NAMESPACE_DECL

#endif // LIBC_TYPES_HAS_FLOAT16

#endif // LLVM_LIBC_SHARED_MATH_FREXPF16_H
