//===-- Implementation of fdimbf16 function -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/math/fdimbf16.h"
#include "src/__support/math/fdimbf16.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(bfloat16, fdimbf16, (bfloat16 x, bfloat16 y)) {
  return math::fdimbf16(x, y);
}

} // namespace LIBC_NAMESPACE_DECL
