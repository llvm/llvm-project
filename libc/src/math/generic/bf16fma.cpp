//===-- Implementation of bf16fma function --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/math/bf16fma.h"
#include "src/__support/math/bf16fma.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(bfloat16, bf16fma, (double x, double y, double z)) {
  return math::bf16fma(x, y, z);
}

} // namespace LIBC_NAMESPACE_DECL
