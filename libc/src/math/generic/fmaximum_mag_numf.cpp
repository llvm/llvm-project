//===-- Implementation of fmaximum_mag_numf function-----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/math/fmaximum_mag_numf.h"
#include "src/__support/math/fmaximum_mag_numf.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(float, fmaximum_mag_numf, (float x, float y)) {
  return math::fmaximum_mag_numf(x, y);
}

} // namespace LIBC_NAMESPACE_DECL
