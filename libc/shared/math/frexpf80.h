//===-- Shared header for frexpf80 ------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SHARED_MATH_FREXPF80_H
#define LLVM_LIBC_SHARED_MATH_FREXPF80_H

#include "src/__support/math/frexpf80.h"

namespace LIBC_NAMESPACE_DECL {

namespace shared {

using math::frexpf80;

} // namespace shared

} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SHARED_MATH_FREXPF80_H
