//===-- Implementation of nanl function -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/math/nanl.h"
#include "src/__support/math/nanl.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(long double, nanl, (const char *arg)) {
  return math::nanl(arg);
}

} // namespace LIBC_NAMESPACE_DECL
