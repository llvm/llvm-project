//===-- Double-precision cosh implementation ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/math/cosh.h"
#include "src/__support/math/cosh.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(double, cosh, (double x)) { return math::cosh(x); }

} // namespace LIBC_NAMESPACE_DECL
