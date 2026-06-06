//===-- Implementation of nextupbf16 function -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/math/nextupbf16.h"
#include "src/__support/math/nextupbf16.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(bfloat16, nextupbf16, (bfloat16 x)) {
  return math::nextupbf16(x);
}

} // namespace LIBC_NAMESPACE_DECL
