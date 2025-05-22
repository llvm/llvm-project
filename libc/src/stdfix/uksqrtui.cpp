//===-- Implementation of uksqrtui function -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "uksqrtui.h"
#include "src/__support/common.h"
#include "src/__support/fixed_point/sqrt.h"

namespace LIBC_NAMESPACE {

LLVM_LIBC_FUNCTION(unsigned accum, uksqrtui, (unsigned int x)) {
#ifdef LIBC_FAST_MATH
  return fixed_point::isqrt_fast(x);
#else
  return fixed_point::isqrt(x);
#endif
}

} // namespace LIBC_NAMESPACE
