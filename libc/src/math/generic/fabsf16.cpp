//===-- Implementation of fabsf16 function --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/math/fabsf16.h"
#include "src/__support/math/fabsf16.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(float16, fabsf16, (float16 x)) { return math::fabsf16(x); }

} // namespace LIBC_NAMESPACE_DECL
