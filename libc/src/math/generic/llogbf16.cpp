//===-- Implementation of llogbf16 function -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/math/llogbf16.h"
#include "src/__support/math/llogbf16.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(long, llogbf16, (float16 x)) { return math::llogbf16(x); }

} // namespace LIBC_NAMESPACE_DECL
