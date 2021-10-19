//===-- Implementation of hypot function ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/math/hypot.h"
#include "src/__support/FPUtil/Hypot.h"
#include "src/__support/common.h"

namespace __llvm_libc {

LLVM_LIBC_FUNCTION(double, hypot, (double x, double y)) {
  return __llvm_libc::fputil::hypot(x, y);
}

} // namespace __llvm_libc
