//===-- Implementation of fmodf16 function --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/math/fmodf16.h"
#include "src/__support/math/fmodf16.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(float16, fmodf16, (float16 x, float16 y)) {
  return math::fmodf16(x, y);
}

} // namespace LIBC_NAMESPACE_DECL
