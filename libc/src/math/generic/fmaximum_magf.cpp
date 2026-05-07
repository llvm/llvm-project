//===-- Implementation of fmaximum_magf function---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/math/fmaximum_magf.h"
#include "src/__support/math/fmaximum_magf.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(float, fmaximum_magf, (float x, float y)) {
  return math::fmaximum_magf(x, y);
}

} // namespace LIBC_NAMESPACE_DECL
