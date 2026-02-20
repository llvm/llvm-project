//===-- Implementation of ceilf function ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/math/ceilf.h"
#include "src/__support/math/ceilf.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(float, ceilf, (float x)) { return math::ceilf(x); }

} // namespace LIBC_NAMESPACE_DECL
