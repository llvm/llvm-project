//===-- Double-precision e^x function -------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/math/exp.h"
#include "shared/math/static_rounding/exp.h"
#include "src/__support/math/exp.h"
namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(double, exp, (double x)) {
// TODO: update to follow the new structure (following
// https://github.com/llvm/llvm-project/pull/224735)
#if !defined(LIBC_TARGET_CPU_HAS_FPU_DOUBLE) &&                                \
    defined(LIBC_MATH_HAS_ASSUME_ROUND_NEAREST_ONLY) &&                        \
    defined(LIBC_MATH_HAS_NO_EXCEPT) && defined(LIBC_MATH_HAS_NO_ERRNO)
  return shared::math::static_rounding::exp(x, FE_TONEAREST);
#else
  return math::exp(x);
#endif
}

} // namespace LIBC_NAMESPACE_DECL
