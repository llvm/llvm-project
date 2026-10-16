//===-- Half-precision atan2 function ---------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/math/atan2f16.h"
#include "src/__support/math/atan2f16.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(float16, atan2f16, (float16 y, float16 x)) {
  return math::atan2f16(y, x);
}

} // namespace LIBC_NAMESPACE_DECL
