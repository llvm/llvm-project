//===-- Single-precision sinh function ------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/math/sinhf.h"
#include "src/__support/math/sinhf.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(float, sinhf, (float x)) { return math::sinhf(x); }

} // namespace LIBC_NAMESPACE_DECL
