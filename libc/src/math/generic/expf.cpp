//===-- Single-precision e^x function -------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/math/expf.h"
#include "src/__support/math/expf.h"
#include "src/__support/math/expf_integer_eval.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(float, expf, (float x)) {
#if defined(LIBC_MATH_HAS_SKIP_ACCURATE_PASS) &&                               \
    defined(LIBC_MATH_SMALL_TABLES) &&                                         \
    !defined(LIBC_TARGET_CPU_HAS_FPU_DOUBLE)
  return math::integer_only::expf(x);
#else
  return math::expf(x);
#endif
}

} // namespace LIBC_NAMESPACE_DECL
