//===-- Implementation for tanbf16(x) function ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception.
//
//===----------------------------------------------------------------------===//

#include "src/math/tanbf16.h"
#include "src/__support/FPUtil/bfloat16.h"
#include "src/__support/math/tanbf16.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(bfloat16, tanbf16, (bfloat16 x)) { return math::tanbf16(x); }

} // namespace LIBC_NAMESPACE_DECL
