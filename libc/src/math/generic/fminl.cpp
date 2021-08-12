//===-- Implementation of fminl function ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/math/fminl.h"
#include "src/__support/FPUtil/BasicOperations.h"
#include "src/__support/common.h"

namespace __llvm_libc {

LLVM_LIBC_FUNCTION(long double, fminl, (long double x, long double y)) {
  return fputil::fmin(x, y);
}

} // namespace __llvm_libc
