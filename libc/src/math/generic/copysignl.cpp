//===-- Implementation of copysignl function ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/math/copysignl.h"
#include "src/__support/FPUtil/ManipulationFunctions.h"
#include "src/__support/common.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

// TODO: Change this implementation to copysignf80 and add long double alias
// similar to copysign and copysignf128.
#if !defined(LIBC_ALIAS_LONG_DOUBLE_TO_DOUBLE) &&                              \
    !defined(LIBC_ALIAS_LONG_DOUBLE_TO_FLOAT128)

LLVM_LIBC_FUNCTION(long double, copysignl, (long double x, long double y)) {
  return fputil::copysign(x, y);
}

#endif

} // namespace LIBC_NAMESPACE_DECL
