//===-- Implementation of fminimum_mag_num function------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/math/fminimum_mag_num.h"
#include "src/__support/FPUtil/BasicOperations.h"
#include "src/__support/common.h"

namespace LIBC_NAMESPACE {

LLVM_LIBC_FUNCTION(double, fminimum_mag_num, (double x, double y)) {
  return fputil::fminimum_mag_num(x, y);
}

} // namespace LIBC_NAMESPACE
