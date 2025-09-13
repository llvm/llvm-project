//===-- Implementation of cbrt function -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/math/cbrt.h"
#include "src/__support/math/cbrt.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(double, cbrt, (double x)) { return math::cbrt(x); }

} // namespace LIBC_NAMESPACE_DECL
