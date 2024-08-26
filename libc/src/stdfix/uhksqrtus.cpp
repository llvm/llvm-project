//===-- Implementation of uhksqrtus function ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "uhksqrtus.h"
#include "src/__support/common.h"
#include "src/__support/fixed_point/sqrt.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(unsigned short accum, uhksqrtus, (unsigned short x)) {
#ifdef LIBC_FAST_MATH
  return fixed_point::isqrt_fast(x);
#else
  return fixed_point::isqrt(x);
#endif
}

} // namespace LIBC_NAMESPACE_DECL
