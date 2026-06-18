//===-- Half-precision rsqrt function -------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception.
//
//===----------------------------------------------------------------------===//

#include "src/math/rsqrtf16.h"
#include "src/__support/math/rsqrtf16.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(float16, rsqrtf16, (float16 x)) { return math::rsqrtf16(x); }
} // namespace LIBC_NAMESPACE_DECL
