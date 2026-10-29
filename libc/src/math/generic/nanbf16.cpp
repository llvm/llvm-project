//===-- Implementation of nanbf16 function --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/math/nanbf16.h"
#include "src/__support/math/nanbf16.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(bfloat16, nanbf16, (const char *arg)) {
  return math::nanbf16(arg);
}

} // namespace LIBC_NAMESPACE_DECL
