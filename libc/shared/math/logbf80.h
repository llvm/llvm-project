//===-- Shared header for logbf80 -------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SHARED_MATH_LOGBF80_H
#define LLVM_LIBC_SHARED_MATH_LOGBF80_H

#include "src/__support/math/logbf80.h"

namespace LIBC_NAMESPACE_DECL {

namespace shared {

using math::logbf80;

} // namespace shared

} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SHARED_MATH_LOGBF80_H
