//===-- Double-precision acos function ------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/math/acos.h"
#include "src/__support/math/acos.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(double, acos, (double x)) { return math::acos(x); }

} // namespace LIBC_NAMESPACE_DECL
