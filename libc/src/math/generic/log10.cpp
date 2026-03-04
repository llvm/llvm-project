//===-- Double-precision log10(x) function --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/math/log10.h"
#include "src/__support/math/log10.h"

namespace LIBC_NAMESPACE_DECL {
LLVM_LIBC_FUNCTION(double, log10, (double x)) { return math::log10(x); }

} // namespace LIBC_NAMESPACE_DECL
