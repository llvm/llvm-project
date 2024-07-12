//===-- Implementation of fminimumf16 function ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/math/fminimumf16.h"
#include "src/__support/FPUtil/BasicOperations.h"
#include "src/__support/common.h"

namespace LIBC_NAMESPACE {

LLVM_LIBC_FUNCTION(float16, fminimumf16, (float16 x, float16 y)) {
  return fputil::fminimum(x, y);
}

} // namespace LIBC_NAMESPACE
