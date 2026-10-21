//===-- Implementation of f16sqrtf128 function ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/math/f16sqrtf128.h"
#include "src/__support/math/f16sqrtf128.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(float16, f16sqrtf128, (float128 x)) {
  return math::f16sqrtf128(x);
}

} // namespace LIBC_NAMESPACE_DECL
