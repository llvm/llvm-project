//===-- Implementation of round function ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/math/round.h"
#include "src/__support/math/round.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(double, round, (double x)) { return math::round(x); }

} // namespace LIBC_NAMESPACE_DECL
