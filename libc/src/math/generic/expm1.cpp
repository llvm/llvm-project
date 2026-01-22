//===-- Double-precision e^x - 1 function ---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/math/expm1.h"
#include "src/__support/math/expm1.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(double, expm1, (double x)) { return math::expm1(x); }

} // namespace LIBC_NAMESPACE_DECL
