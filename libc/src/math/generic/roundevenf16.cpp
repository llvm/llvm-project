//===-- Implementation of roundevenf16 function ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/math/roundevenf16.h"
#include "src/__support/FPUtil/NearestIntegerOperations.h"
#include "src/__support/common.h"

namespace LIBC_NAMESPACE {

LLVM_LIBC_FUNCTION(float16, roundevenf16, (float16 x)) {
  return fputil::round_using_specific_rounding_mode(x, FP_INT_TONEAREST);
}

} // namespace LIBC_NAMESPACE
