//===-- Implementation of fmaximum_mag_numl function-----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/math/fmaximum_mag_numl.h"
#include "src/__support/math/fmaximum_mag_numl.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(long double, fmaximum_mag_numl,
                   (long double x, long double y)) {
  return math::fmaximum_mag_numl(x, y);
}

} // namespace LIBC_NAMESPACE_DECL
