//===-- Implementation of nexttowardf function ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/math/nexttowardf.h"
#include "src/__support/math/nexttowardf.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(float, nexttowardf, (float x, long double y)) {
  return math::nexttowardf(x, y);
}

} // namespace LIBC_NAMESPACE_DECL
