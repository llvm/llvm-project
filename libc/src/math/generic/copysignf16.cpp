//===-- Implementation of copysignf16 function ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/math/copysignf16.h"
#include "src/__support/FPUtil/ManipulationFunctions.h"
#include "src/__support/common.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(float16, copysignf16, (float16 x, float16 y)) {
#ifdef __LIBC_MISC_MATH_BASIC_OPS_OPT
  return __builtin_copysignf16(x, y);
#else
  return fputil::copysign(x, y);
#endif
}

} // namespace LIBC_NAMESPACE_DECL
