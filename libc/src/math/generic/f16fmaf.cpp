//===-- Implementation of f16fmaf function --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/math/f16fmaf.h"
#include "src/__support/math/f16fmaf.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(float16, f16fmaf, (float x, float y, float z)) {
  return math::f16fmaf(x, y, z);
}

} // namespace LIBC_NAMESPACE_DECL
