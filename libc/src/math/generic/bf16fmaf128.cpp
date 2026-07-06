//===-- Implementation of bf16fmaf128 function ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/math/bf16fmaf128.h"
#include "src/__support/math/bf16fmaf128.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(bfloat16, bf16fmaf128,
                   (float128 x, float128 y, float128 z)) {
  return math::bf16fmaf128(x, y, z);
}

} // namespace LIBC_NAMESPACE_DECL
