//===-- Double-precision asin function ------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/math/asin.h"
#include "src/__support/math/asin.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(double, asin, (double x)) { return math::asin(x); }

} // namespace LIBC_NAMESPACE_DECL
