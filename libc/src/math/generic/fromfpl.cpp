//===-- Implementation of fromfpl function --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/math/fromfpl.h"
#include "src/__support/FPUtil/NearestIntegerOperations.h"
#include "src/__support/common.h"

namespace LIBC_NAMESPACE {

LLVM_LIBC_FUNCTION(long double, fromfpl,
                   (long double x, int rnd, unsigned int width)) {
  return fputil::fromfp</*IsSigned=*/true>(x, rnd, width);
}

} // namespace LIBC_NAMESPACE
