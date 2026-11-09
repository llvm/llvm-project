//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \brief
/// This file will be used for shared_math_test.cpp and
/// shared_math_constexpr_test.cpp
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SHARED_MATH_ISNANBF16_H
#define LLVM_LIBC_SHARED_MATH_ISNANBF16_H

#include "shared/libc_common.h"
#include "src/__support/math/isnanbf16.h"

namespace LIBC_NAMESPACE_DECL {
namespace shared {
using math::isnanbf16;
} // namespace shared
} // namespace LIBC_NAMESPACE_DECL
#endif // LLVM_LIBC_SHARED_MATH_ISNANBF16_H
