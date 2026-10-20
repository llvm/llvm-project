//===-- Implementation of exp10m1f function -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/math/exp10m1f.h"
#include "src/__support/math/exp10m1f.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(float, exp10m1f, (float x)) { return math::exp10m1f(x); }

} // namespace LIBC_NAMESPACE_DECL
