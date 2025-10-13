//===-- Half-precision atanh(x) function ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/math/atanhf16.h"
#include "src/__support/math/atanhf16.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(float16, atanhf16, (float16 x)) { return math::atanhf16(x); }

} // namespace LIBC_NAMESPACE_DECL
