//===-- Implementation of the GPU roundl function -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/math/roundl.h"
#include "src/__support/FPUtil/PlatformDefs.h"
#include "src/__support/common.h"

namespace __llvm_libc {

#ifndef LONG_DOUBLE_IS_DOUBLE
#error "GPU targets do not support long doubles"
#endif

LLVM_LIBC_FUNCTION(long double, roundl, (long double x)) {
  return __builtin_round(x);
}

} // namespace __llvm_libc
