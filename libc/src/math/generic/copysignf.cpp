//===-- Implementation of copysignf function ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/math/copysignf.h"
#include "src/__support/FPUtil/ManipulationFunctions.h"
#include "src/__support/common.h"

namespace __llvm_libc {

LLVM_LIBC_FUNCTION(float, copysignf, (float x, float y)) {
  return fputil::copysign(x, y);
}

} // namespace __llvm_libc
