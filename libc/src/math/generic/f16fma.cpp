//===-- Implementation of f16fma function ---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/math/f16fma.h"
#include "src/__support/math/f16fma.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(float16, f16fma, (double x, double y, double z)) {
  return math::f16fma(x, y, z);
}

} // namespace LIBC_NAMESPACE_DECL
