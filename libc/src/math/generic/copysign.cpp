//===-- Implementation of copysign function -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/math/copysign.h"
#include "src/__support/FPUtil/ManipulationFunctions.h"
#include "src/__support/common.h"
#include "src/__support/macros/config.h"
#include "src/__support/macros/properties/types.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(double, copysign, (double x, double y)) {
#ifdef __LIBC_MISC_MATH_BASIC_OPS_OPT
  return __builtin_copysign(x, y);
#else
  return fputil::copysign(x, y);
#endif
}

} // namespace LIBC_NAMESPACE_DECL

#if defined(LIBC_TYPES_LONG_DOUBLE_IS_FLOAT64) &&                              \
    defined(LIBC_ALIAS_LONG_DOUBLE)
#include "src/math/copysignl.h"

namespace LIBC_NAMESPACE_DECL {
decltype(LIBC_NAMESPACE::copysignl) copysignl [[gnu::alias("copysignl")]];
asm("copysignl = copysign");
} // namespace LIBC_NAMESPACE_DECL

#endif // LIBC_TYPES_LONG_DOUBLE_IS_FLOAT64