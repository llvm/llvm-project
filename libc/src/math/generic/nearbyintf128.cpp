//===-- Implementation of nearbyintf128 function --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/math/nearbyintf128.h"
#include "src/__support/FPUtil/NearestIntegerOperations.h"
#include "src/__support/common.h"

namespace LIBC_NAMESPACE {

LLVM_LIBC_FUNCTION(float128, nearbyintf128, (float128 x)) {
  return fputil::round_using_current_rounding_mode(x);
}

} // namespace LIBC_NAMESPACE
