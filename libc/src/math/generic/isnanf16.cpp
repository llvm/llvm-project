//===-- Implementation of isnanf16 function -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/math/isnanf16.h"
#include "src/__support/math/isnanf16.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(int, isnanf16, (float16 x)) { return math::isnanf16(x); }

} // namespace LIBC_NAMESPACE_DECL
