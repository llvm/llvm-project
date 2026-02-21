//===-- Double-precision tan function -------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/math/tan.h"
#include "src/__support/math/tan.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(double, tan, (double x)) { return math::tan(x); }

} // namespace LIBC_NAMESPACE_DECL
