//===-- Implementation of ceill function ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/math/ceill.h"
#include "src/__support/math/ceill.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(long double, ceill, (long double x)) {
  return math::ceill(x);
}

} // namespace LIBC_NAMESPACE_DECL
