//===-- Implementation of fmaximum_magl function---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/math/fmaximum_magl.h"
#include "src/__support/FPUtil/BasicOperations.h"
#include "src/__support/common.h"

namespace LIBC_NAMESPACE {

LLVM_LIBC_FUNCTION(long double, fmaximum_magl, (long double x, long double y)) {
  return fputil::fmaximum_mag(x, y);
}

} // namespace LIBC_NAMESPACE
