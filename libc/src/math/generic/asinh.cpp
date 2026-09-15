//===-- Double-precision asinh implementation -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/math/asinh.h"
#include "src/__support/math/asinh.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(double, asinh, (double x)) { return math::asinh(x); }

} // namespace LIBC_NAMESPACE_DECL
