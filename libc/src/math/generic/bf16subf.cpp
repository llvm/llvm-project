//===-- Implementation of bf16subf function -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/math/bf16subf.h"
#include "src/__support/math/bf16subf.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(bfloat16, bf16subf, (float x, float y)) {
  return math::bf16subf(x, y);
}

} // namespace LIBC_NAMESPACE_DECL
