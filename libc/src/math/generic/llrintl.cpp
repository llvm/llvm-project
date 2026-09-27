//===-- Implementation of llrintl function --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/math/llrintl.h"
#include "src/__support/math/llrintl.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(long long, llrintl, (long double x)) {
  return math::llrintl(x);
}

} // namespace LIBC_NAMESPACE_DECL
