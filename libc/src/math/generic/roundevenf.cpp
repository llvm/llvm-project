//===-- Implementation of roundevenf function -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/math/roundevenf.h"
#include "src/__support/math/roundevenf.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(float, roundevenf, (float x)) { return math::roundevenf(x); }

} // namespace LIBC_NAMESPACE_DECL
