//===-- Implementation of fabs function -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/math/fabs.h"
#include "src/__support/FPUtil/BasicOperations.h"
#include "src/__support/common.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(double, fabs, (double x)) {
#ifdef __LIBC_MISC_MATH_BASIC_OPS_OPT
  return __builtin_fabs(x);
#else
  return fputil::abs(x);
#endif
}

} // namespace LIBC_NAMESPACE_DECL
