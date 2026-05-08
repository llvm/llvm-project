//===-- Implementation of trunc function ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/math/trunc.h"
#include "src/__support/math/trunc.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(double, trunc, (double x)) { return math::trunc(x); }

} // namespace LIBC_NAMESPACE_DECL
