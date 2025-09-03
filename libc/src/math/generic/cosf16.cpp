//===-- Half-precision cos(x) function ------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/math/cosf16.h"
#include "src/__support/math/cosf16.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(float16, cosf16, (float16 x)) { return math::cosf16(x); }

} // namespace LIBC_NAMESPACE_DECL
