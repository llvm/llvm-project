//===-- Implementation of nexttowardf16 function --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/math/nexttowardf16.h"
#include "src/__support/math/nexttowardf16.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(float16, nexttowardf16, (float16 x, long double y)) {
  return math::nexttowardf16(x, y);
}

} // namespace LIBC_NAMESPACE_DECL
