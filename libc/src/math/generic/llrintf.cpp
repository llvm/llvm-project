//===-- Implementation of llrintf function --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/math/llrintf.h"
#include "src/__support/math/llrintf.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(long long, llrintf, (float x)) { return math::llrintf(x); }

} // namespace LIBC_NAMESPACE_DECL
