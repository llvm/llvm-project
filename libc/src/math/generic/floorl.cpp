//===-- Implementation of floorl function ---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/math/floorl.h"
#include "src/__support/FPUtil/NearestIntegerOperations.h"
#include "src/__support/common.h"

namespace __llvm_libc {

LLVM_LIBC_FUNCTION(long double, floorl, (long double x)) {
  return fputil::floor(x);
}

} // namespace __llvm_libc
