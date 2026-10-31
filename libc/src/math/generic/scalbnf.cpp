//===-- Single-precision scalbnf function ---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/math/scalbnf.h"
#include "src/__support/math/scalbnf.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(float, scalbnf, (float x, int n)) {
  return math::scalbnf(x, n);
}

} // namespace LIBC_NAMESPACE_DECL
