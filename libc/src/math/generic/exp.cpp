//===-- Double-precision e^x function -------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/math/exp.h"
#include "src/__support/math/exp.h"
namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(double, exp, (double x)) { return math::exp(x); }

} // namespace LIBC_NAMESPACE_DECL
