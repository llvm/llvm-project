//===-- Implementation of daddl function ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/math/daddl.h"
#include "src/__support/FPUtil/generic/add_sub.h"
#include "src/__support/common.h"
#include "src/__support/macros/config.h"
#include "src/__support/macros/properties/types.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(double, daddl, (long double x, long double y)) {
#ifdef LIBC_TYPES_LONG_DOUBLE_IS_FLOAT64
  return static_cast<double>(x) + static_cast<double>(y);
#else
  return fputil::generic::add<double>(x, y);
#endif
}

} // namespace LIBC_NAMESPACE_DECL
