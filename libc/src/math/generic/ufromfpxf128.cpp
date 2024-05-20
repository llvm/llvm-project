//===-- Implementation of ufromfpxf128 function ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/math/ufromfpxf128.h"
#include "src/__support/FPUtil/NearestIntegerOperations.h"
#include "src/__support/common.h"

namespace LIBC_NAMESPACE {

LLVM_LIBC_FUNCTION(float128, ufromfpxf128,
                   (float128 x, int rnd, unsigned int width)) {
  return fputil::fromfpx</*IsSigned=*/false>(x, rnd, width);
}

} // namespace LIBC_NAMESPACE
