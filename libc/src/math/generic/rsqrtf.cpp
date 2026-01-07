//===-- Single-precision rsqrt function -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception.
//
//===----------------------------------------------------------------------===//

#include "src/math/rsqrtf.h"
#include "src/__support/math/rsqrtf.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(float, rsqrtf, (float x)) { return math::rsqrtf(x); }

} // namespace LIBC_NAMESPACE_DECL
