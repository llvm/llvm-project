//===-- Implementation of lrintf128 function ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/math/lrintf128.h"
#include "src/__support/math/lrintf128.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(long, lrintf128, (float128 x)) { return math::lrintf128(x); }

} // namespace LIBC_NAMESPACE_DECL
