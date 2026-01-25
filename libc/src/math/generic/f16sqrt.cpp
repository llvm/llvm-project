//===-- Implementation of f16sqrt function --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/math/f16sqrt.h"
#include "src/__support/math/f16sqrt.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(float16, f16sqrt, (double x)) { return math::f16sqrt(x); }

} // namespace LIBC_NAMESPACE_DECL
