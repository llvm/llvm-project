//===-- Implementation of dmull function ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/math/dmull.h"
#include "src/__support/FPUtil/generic/mul.h"
#include "src/__support/common.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(double, dmull, (long double x, long double y)) {
  return fputil::generic::mul<double>(x, y);
}

} // namespace LIBC_NAMESPACE_DECL
