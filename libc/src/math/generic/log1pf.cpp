//===-- Single-precision log1p(x) function --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/math/log1pf.h"
#include "src/__support/math/log1pf.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(float, log1pf, (float x)) { return math::log1pf(x); }

} // namespace LIBC_NAMESPACE_DECL
