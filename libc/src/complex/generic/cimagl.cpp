//===-- Implementation of cimagl function ---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/complex/cimagl.h"
#include "src/__support/FPUtil/NearestIntegerOperations.h"
#include "src/__support/common.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(long double, cimagl, (_Complex long double x)) {
  long double *xCmplxPtr = reinterpret_cast<long double *>(&x);
  return xCmplxPtr[1];
}

} // namespace LIBC_NAMESPACE_DECL
