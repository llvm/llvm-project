//===-- Implementation of ilogbl function ---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/math/ilogbl.h"
#include "src/__support/math/ilogbl.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(int, ilogbl, (long double x)) { return math::ilogbl(x); }

} // namespace LIBC_NAMESPACE_DECL
