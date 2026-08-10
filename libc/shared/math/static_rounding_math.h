//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains the statically rounded implementations of math functions
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SHARED_MATH_STATIC_ROUNDING_MATH_H
#define LLVM_LIBC_SHARED_MATH_STATIC_ROUNDING_MATH_H

#include "shared/libc_common.h"
#include "static_rounding/expf.h"

namespace LIBC_NAMESPACE_DECL {

namespace shared {

using math::static_rounding::expf;

} // namespace shared

} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SHARED_MATH_STATIC_ROUNDING_MATH_H
