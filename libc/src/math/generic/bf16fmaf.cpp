//===-- Implementation of bf16fmaf function -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/math/bf16fmaf.h"
#include "src/__support/math/bf16fmaf.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(bfloat16, bf16fmaf, (float x, float y, float z)) {
  return math::bf16fmaf(x, y, z);
}

} // namespace LIBC_NAMESPACE_DECL
