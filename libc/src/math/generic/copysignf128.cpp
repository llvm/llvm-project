//===-- Implementation of copysignf128 function ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/math/copysignf128.h"
#include "src/__support/FPUtil/ManipulationFunctions.h"
#include "src/__support/common.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(float128, copysignf128, (float128 x, float128 y)) {
  return fputil::copysign(x, y);
}

} // namespace LIBC_NAMESPACE_DECL

#if defined(LIBC_ALIAS_LONG_DOUBLE_TO_FLOAT128)
#include "src/math/copysignl.h"

LLVM_LIBC_ALIAS(copysignl, copysignf128);

#endif // LIBC_TYPES_LONG_DOUBLE_IS_FLOAT128
