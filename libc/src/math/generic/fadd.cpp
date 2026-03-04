//===-- Implementation of fadd function -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/math/fadd.h"
#include "src/__support/math/fadd.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(float, fadd, (double x, double y)) {
  return math::fadd(x, y);
}

} // namespace LIBC_NAMESPACE_DECL
