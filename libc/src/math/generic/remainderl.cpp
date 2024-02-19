//===-- Implementation of remainderl function -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/math/remainderl.h"
#include "src/__support/FPUtil/DivisionAndRemainderOperations.h"
#include "src/__support/common.h"

namespace LIBC_NAMESPACE {

LLVM_LIBC_FUNCTION(long double, remainderl, (long double x, long double y)) {
  int quotient;
  return fputil::remquo(x, y, quotient);
}

} // namespace LIBC_NAMESPACE
