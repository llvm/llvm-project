//===-- Single-precision erfc function ------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/math/erfcf.h"
#include "src/__support/common.h"
#include "src/__support/math/erfcf.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(float, erfcf, (float x)) { return math::erfcf(x); }

} // namespace LIBC_NAMESPACE_DECL
