//===-- Implementation of the fminimum_mag_numf function for GPU-----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/math/fminimum_mag_numf.h"

#include "src/__support/CPP/bit.h"
#include "src/__support/common.h"
#include "src/__support/macros/optimization.h"

namespace LIBC_NAMESPACE {

LLVM_LIBC_FUNCTION(float, fminimum_mag_numf, (float x, float y)) {
  return __builtin_fminimum_mag_num(x, y);
}

} // namespace LIBC_NAMESPACE
