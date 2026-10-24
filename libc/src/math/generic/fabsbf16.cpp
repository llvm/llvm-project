//===-- Implementation of fabsbf16 function -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/math/fabsbf16.h"
#include "src/__support/math/fabsbf16.h"

namespace LIBC_NAMESPACE_DECL {

#if defined(LIBC_TYPES_HAS_BUILTIN_BFLOAT16)
LLVM_LIBC_FUNCTION(__bf16, fabsbf16, (__bf16 x)) { return math::fabsbf16(x); }
#else
LLVM_LIBC_FUNCTION(bfloat16, fabsbf16, (bfloat16 x)) {
  return math::fabsbf16(x);
}
#endif

} // namespace LIBC_NAMESPACE_DECL
