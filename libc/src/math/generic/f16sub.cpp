//===-- Implementation of f16sub function ---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/math/f16sub.h"
#include "src/__support/math/f16sub.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(float16, f16sub, (double x, double y)) {
  return math::f16sub(x, y);
}

} // namespace LIBC_NAMESPACE_DECL
