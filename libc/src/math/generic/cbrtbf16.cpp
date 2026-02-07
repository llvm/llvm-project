//===-- Implementation of cbrtbf16 function -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/math/cbrtbf16.h"
#include "src/__support/FPUtil/bfloat16.h"
#include "src/__support/macros/config.h"
#include "src/__support/math/cbrtf.h"

namespace LIBC_NAMESPACE_DECL {
LLVM_LIBC_FUNCTION(bfloat16, cbrtbf16, (bfloat16 x)) {
  return static_cast<bfloat16>(math::cbrtf(static_cast<float>(x)));
}
} // namespace LIBC_NAMESPACE_DECL
