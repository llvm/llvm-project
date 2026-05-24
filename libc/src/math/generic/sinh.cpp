//===-- Double-precision sinh implementation ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/math/sinh.h"
#include "src/__support/math/sinh.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(double, sinh, (double x)) { return math::sinh(x); }

} // namespace LIBC_NAMESPACE_DECL
