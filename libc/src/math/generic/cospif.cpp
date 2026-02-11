//===-- Single-precision cospi function -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/math/cospif.h"
#include "src/__support/math/cospif.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(float, cospif, (float x)) { return math::cospif(x); }

} // namespace LIBC_NAMESPACE_DECL
