//===-- Implementation of fdim function -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/math/fdim.h"
#include "src/__support/math/fdim.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(double, fdim, (double x, double y)) {
  return math::fdim(x, y);
}

} // namespace LIBC_NAMESPACE_DECL
