//===-- Implementation of fmaximum_num function----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/math/fmaximum_num.h"
#include "src/__support/math/fmaximum_num.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(double, fmaximum_num, (double x, double y)) {
  return math::fmaximum_num(x, y);
}

} // namespace LIBC_NAMESPACE_DECL
