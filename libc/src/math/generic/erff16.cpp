//===-- Half-precision erf(x) function ------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/math/erff16.h"
#include "src/__support/math/erff16.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(float16, erff16, (float16 x)) { return math::erff16(x); }

} // namespace LIBC_NAMESPACE_DECL
