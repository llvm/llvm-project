//===-- Single-precision acosh function -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/math/acoshf.h"

#include "src/__support/math/acoshf.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(float, acoshf, (float x)) { return math::acoshf(x); }

} // namespace LIBC_NAMESPACE_DECL
