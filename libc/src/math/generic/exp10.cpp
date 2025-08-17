//===-- Double-precision 10^x function ------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/math/exp10.h"
#include "src/__support/math/exp10.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(double, exp10, (double x)) { return math::exp10(x); }

} // namespace LIBC_NAMESPACE_DECL
