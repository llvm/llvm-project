//===-- Implementation of fmaximum_mag_numf128 function--------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/math/fmaximum_mag_numf128.h"
#include "src/__support/FPUtil/BasicOperations.h"
#include "src/__support/common.h"

namespace LIBC_NAMESPACE {

LLVM_LIBC_FUNCTION(float128, fmaximum_mag_numf128, (float128 x, float128 y)) {
  return fputil::fmaximum_mag_num(x, y);
}

} // namespace LIBC_NAMESPACE
