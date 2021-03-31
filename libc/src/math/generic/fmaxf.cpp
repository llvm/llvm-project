//===-- Implementation of fmaxf function ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/math/fmaxf.h"
#include "src/__support/common.h"
#include "utils/FPUtil/BasicOperations.h"

namespace __llvm_libc {

LLVM_LIBC_FUNCTION(float, fmaxf, (float x, float y)) {
  return fputil::fmax(x, y);
}

} // namespace __llvm_libc
