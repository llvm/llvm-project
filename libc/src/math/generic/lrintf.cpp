//===-- Implementation of lrintf function ---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/math/lrintf.h"
#include "src/__support/math/lrintf.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(long, lrintf, (float x)) { return math::lrintf(x); }

} // namespace LIBC_NAMESPACE_DECL
