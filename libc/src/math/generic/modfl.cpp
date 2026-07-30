//===-- Implementation of modfl function ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/math/modfl.h"
#include "src/__support/math/modfl.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(long double, modfl, (long double x, long double *iptr)) {
  return math::modfl(x, iptr);
}

} // namespace LIBC_NAMESPACE_DECL
