//===-- Implementation of frexp function ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/math/frexp.h"
#include "src/__support/FPUtil/ManipulationFunctions.h"
#include "src/__support/common.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(double, frexp, (double x, int *exp)) {
  return fputil::frexp(x, *exp);
}

} // namespace LIBC_NAMESPACE_DECL
