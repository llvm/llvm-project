//===-- Implementation of nexttowardl function ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/math/nexttowardl.h"
#include "src/__support/FPUtil/ManipulationFunctions.h"
#include "src/__support/common.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(long double, nexttowardl, (long double x, long double y)) {
  // We can reuse the nextafter implementation because the internal nextafter is
  // templated on the types of the arguments.
  return fputil::nextafter(x, y);
}

} // namespace LIBC_NAMESPACE_DECL
