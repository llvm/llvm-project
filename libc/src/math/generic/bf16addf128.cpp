//===-- Implementation of bf16addf128 function ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/math/bf16addf128.h"
#include "src/__support/math/bf16addf128.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(bfloat16, bf16addf128, (float128 x, float128 y)) {
  return math::bf16addf128(x, y);
}

} // namespace LIBC_NAMESPACE_DECL
