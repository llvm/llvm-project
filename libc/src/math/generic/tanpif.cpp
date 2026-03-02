//===-- Single-precision tanpi function -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/math/tanpif.h"
#include "src/__support/math/tanpif.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(float, tanpif, (float x)) { return math::tanpif(x); }

} // namespace LIBC_NAMESPACE_DECL
