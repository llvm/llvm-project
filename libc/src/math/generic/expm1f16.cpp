//===-- Half-precision e^x - 1 function -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/math/expm1f16.h"
#include "src/__support/math/expm1f16.h"

namespace LIBC_NAMESPACE_DECL {
LLVM_LIBC_FUNCTION(float16, expm1f16, (float16 x)) { return math::expm1f16(x); }

} // namespace LIBC_NAMESPACE_DECL
