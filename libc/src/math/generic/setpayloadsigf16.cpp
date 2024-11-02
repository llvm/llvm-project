//===-- Implementation of setpayloadsigf16 function -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/math/setpayloadsigf16.h"
#include "src/__support/FPUtil/BasicOperations.h"
#include "src/__support/common.h"

namespace LIBC_NAMESPACE {

LLVM_LIBC_FUNCTION(int, setpayloadsigf16, (float16 * res, float16 pl)) {
  return static_cast<int>(fputil::setpayload</*IsSignaling=*/true>(*res, pl));
}

} // namespace LIBC_NAMESPACE
