//===-- Implementation of copysignf function ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/math/copysignf.h"
#include "src/__support/FPUtil/ManipulationFunctions.h"
#include "src/__support/common.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(float, copysignf, (float x, float y)) {
#ifdef __LIBC_MISC_MATH_BASIC_OPS_OPT
  return __builtin_copysignf(x, y);
#else
  return fputil::copysign(x, y);
#endif
}

} // namespace LIBC_NAMESPACE_DECL
