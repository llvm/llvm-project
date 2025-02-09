//===-- Implementation of rintf function ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/math/rintf.h"
#include "src/__support/FPUtil/NearestIntegerOperations.h"
#include "src/__support/common.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(float, rintf, (float x)) {
#ifdef __LIBC_USE_BUILTIN_CEIL_FLOOR_RINT_TRUNC
  return __builtin_rintf(x);
#else
  return fputil::round_using_current_rounding_mode(x);
#endif
}

} // namespace LIBC_NAMESPACE_DECL
