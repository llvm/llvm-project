//===-- Implementation of copysignf16 function ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/math/copysignf16.h"
#include "src/__support/FPUtil/ManipulationFunctions.h"
#include "src/__support/common.h"

namespace LIBC_NAMESPACE {

LLVM_LIBC_FUNCTION(float16, copysignf16, (float16 x, float16 y)) {
  return fputil::copysign(x, y);
}

} // namespace LIBC_NAMESPACE
